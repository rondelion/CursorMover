import os
import json
import argparse
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

show_encode_and_decode = True


class WriterSingleton:
    writer = None
    global_step = 0

    def __init__(self):
        pass

    @staticmethod
    def get_writer():
        if not WriterSingleton.writer:
            WriterSingleton.writer = SummaryWriter()
            print("---------------> Created writer at logdir(): ", WriterSingleton.writer.get_logdir())
        return WriterSingleton.writer


class ImageTransform:
    # mean = (0.5,)
    # std = (0.5,)
    def __init__(self):
        self.data_transform = transforms.Compose([transforms.ToTensor()])  # , transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.data_transform(img)


def train(args, ae, model, device, train_loader, global_step, optimizer):
    """Trains the model for one epoch."""
    model.train()

    for batch_idx, (data) in enumerate(train_loader):
        data = data[0].to(device)
        data = torch.flatten(data, 1)
        ae.learn(data, data, optimizer)
        global_step += 1

        if args.dry_run:
            break

    return global_step


def test(model, device, test_loader, epoch, writer):
    """Evaluates the trained model."""
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_loader):
            data = data[0].to(device)
            shape = data.size()
            data = torch.flatten(data, 1)
            y_hat, z_mean, z_logvar = model(data, None)
            loss_dict = model.loss(data, y_hat, z_mean, z_logvar)
            test_loss += loss_dict["loss"].item()  # sum up batch loss

        writer.add_image('pre-test/inputs', torchvision.utils.make_grid(torch.reshape(data, shape)), epoch)
        writer.add_image('pre-test/outputs', torchvision.utils.make_grid(torch.reshape(y_hat, shape)), epoch)
        test_loss /= len(test_loader.dataset)
        writer.add_scalar('pre-test/avg_loss', test_loss, epoch)
        writer.flush()
        print('Test epoch: {}\tAverage loss: {:.4f}'.format(epoch, test_loss))


def log_dump(model, device, test_loader, epoch, dump_fp):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_loader):
            labels = data[1].tolist()
            data = data[0].to(device)
            data = torch.flatten(data, 1)
            y_hat, z_mean, z_logvar = model(data, None)
            batch_size = len(labels)
            for i in range(batch_size):
                label = test_loader.dataset.classes[labels[i]]
                suit = np.zeros(4, dtype=int)  # one-hot vector
                if int(label[0]) > 0:
                    suit[int(label[0]) - 1] = 1
                cursor = np.zeros(2, dtype=int)  # one-hot vector
                if int(label[1]) > 0:
                    cursor[int(label[1]) - 1] = 1
                borders = ",".join(["{}".format(n) for n in label[2:]])
                dump_fp.write(",".join(["{}".format(n) for n in suit]) + "," +
                              ",".join(["{}".format(n) for n in cursor]) + "," +
                              borders + "," +
                              ",".join(["{:.2f}".format(n) for n in z_mean[i].tolist()]) + "\n")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--config', type=str, default='CursorMover.json', metavar='N',
                        help='Model configuration (default: CursorMover.json')
    parser.add_argument('--config-path', type=str, default='agent/ObjectRecognizer/model_config', metavar='N',
                        help='Model configuration (default: agent/ObjectRecognizer/model_config')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='Batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-dump', help='Log file path; no training if provided')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', help='File path for saving the current Model')
    parser.add_argument('--dataset', help='Input dataset path')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    with open(args.config) as config_file:
        ca_config = json.load(config_file)

    config_path = args.config_path.strip().split('/')
    config = ca_config
    for i in range(len(config_path)):
        if config_path[i] in config:
            config = config[config_path[i]]
        else:
            print("Config doesn't match config path!")
            exit(1)

    kwargs = {'batch_size': args.batch_size, 'shuffle': True}

    if use_cuda:
        kwargs.update({
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        })

    # writer = SummaryWriter()  # for TensorBoard
    writer = WriterSingleton.get_writer()

    images = torchvision.datasets.ImageFolder("env/fovea", transform=ImageTransform())

    train_loader = data.DataLoader(images, **kwargs)
    test_loader = data.DataLoader(images, **kwargs)

    imgs, labels = next(iter(train_loader))

    print("image shape ==>;", imgs[0].shape)

    pic = transforms.ToPILImage(mode='RGB')(imgs[0])
    plt.imshow(pic)
    print("Label is ", images.classes[labels[0].item()])
    plt.show()

    from neu_VED import VariationalEncoderDecoder
    config['device'] = device
    config['input_dim'] = imgs[0].numel()
    config['output_dim'] = imgs[0].numel()
    # shape = imgs.size()[1:]
    ae = VariationalEncoderDecoder(config)
    model = ae.model
    import torch.optim as optim
    optimizer = eval("optim." + config['optimizer'] + "(model.parameters(), lr=config['learning_rate'])")

    if args.save_model is not None and os.path.isfile(args.save_model):
        model.load_state_dict(torch.load(args.save_model))
        model.eval()

    if args.log_dump is None:
        global_step = 0
        for epoch in range(0, args.epochs):
            global_step = train(args, ae, model, device, train_loader, global_step, optimizer)
            WriterSingleton.global_step = epoch
            test(model, device, test_loader, epoch, writer)
        if args.save_model is not None:
            torch.save(model.state_dict(), args.save_model)
    else:   # log_dump mode
        dump_fp = open(args.log_dump, mode='w')
        for epoch in range(0, args.epochs):
            log_dump(model, device, test_loader, epoch, dump_fp)
        dump_fp.close()


if __name__ == '__main__':
    main()
