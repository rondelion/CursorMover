import pygame
import os

class ImageMaker():
    def __init__(self, config):
        self.grid_size = config["grid_size"]
        self.image_path = config["image_path"]
        self.out_path = config["out_path"]
        self.suits = ["club", "heart", "diamond", "spade"]
        self.images = {}
        for i in range(4):
            self.images[i] = pygame.image.load(self.image_path + self.suits[i] + ".jpg")
        self.cursor_colors = ["blue_cursor", "pink_cursor"]
        self.cursors = {}
        for i in range(2):
            self.cursors[i] = pygame.image.load(self.image_path + self.cursor_colors[i] + ".png")

        pygame.init()
        self.clock = pygame.time.Clock()

    def draw_frame(self, grid, pos):
        pos_bin = [0, 0, 0, 0]
        if pos == 0:
            return grid, pos_bin
        elif pos == 1:  # Left
            x = 0
            y = -1
            pos_bin[0] = 1
        elif pos == 2:  # Right
            x = -2
            y = -1
            pos_bin[1] = 1
        elif pos == 3:  # Top
            x = -1
            y = 0
            pos_bin[2] = 1
        elif pos == 4:  # Bottom
            x = -1
            y = -2
            pos_bin[3] = 1
        elif pos == 5:  # Top-Left
            x = 0
            y = 0
            pos_bin[0] = 1
            pos_bin[2] = 1
        elif pos == 6:  # Top-Right
            x = -2
            y = 0
            pos_bin[1] = 1
            pos_bin[2] = 1
        elif pos == 7:  # Bottom-Left
            x = 0
            y = -2
            pos_bin[0] = 1
            pos_bin[3] = 1
        elif pos == 8:           # Bottom-Right
            x = -2
            y = -2
            pos_bin[1] = 1
            pos_bin[3] = 1
        pygame.draw.rect(grid, (100, 100, 100), pygame.Rect(x, y, self.grid_size + 2, self.grid_size + 2), 1)
        return grid, pos_bin

    def make_path(self, i, j, pos_bin):
        label = "{0}{1}{2}{3}{4}{5}".format((i + 1) % 5, (j + 1) % 3, pos_bin[0], pos_bin[1], pos_bin[2], pos_bin[3])
        path = self.out_path + "/" + label
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def step(self):
        for i in range(5):
            i_bin = format(i, 'b')
            for j in range(3):
                j_bin = format(j, 'b')
                for k in range(9):
                    grid = pygame.Surface((self.grid_size, self.grid_size))
                    if i < 4:
                        grid.blit(self.images[i], (0, 0))
                        if j < 2:
                            grid.blit(self.cursors[j], (self.grid_size - 13, self.grid_size - 16))
                            grid, pos_bin = self.draw_frame(grid, k)
                            path = self.make_path(i, j, pos_bin)
                            pygame.image.save(grid, path + "/" + self.suits[i] + "_" + self.cursor_colors[j] + str(k) + ".jpg")
                        else:
                            grid, pos_bin = self.draw_frame(grid, k)
                            path = self.make_path(i, j, pos_bin)
                            pygame.image.save(grid, path + "/" + self.suits[i] + str(k) + ".jpg")
                    else:
                        if j < 2:
                            grid.blit(self.cursors[j], (self.grid_size - 13, self.grid_size - 16))
                            grid, pos_bin = self.draw_frame(grid, k)
                            path = self.make_path(i, j, pos_bin)
                            pygame.image.save(grid, path + "/" + "blank_" + self.cursor_colors[j] + str(k) + ".jpg")
                        else:
                            grid, pos_bin = self.draw_frame(grid, k)
                            path = self.make_path(i, j, pos_bin)
                            pygame.image.save(grid, path + "/" + "blank" + str(k) + ".jpg")


def main():
    config = {
        "grid_size": 28,
        "image_path": "env/28×28/",
        "out_path": "env/fovea"
    }
    im = ImageMaker(config)
    im.step()


if __name__ == '__main__':
    main()
