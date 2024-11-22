import pygame
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from utils import SMB, EnemyType, StaticTileType, ColorMap, DynamicTileType, Item
from typing import Tuple, List, Optional
from yolo_class_mapping import ClassMapping


# TODO GridVisualizer 완성하기
class GridVisualizer(QtWidgets.QWidget):
    def __init__(self, parent, size):
        super().__init__(parent)
        self.size = size
        self.ram = None
        self.x_offset = 150
        self.tile_width, self.tile_height = 20, 20
        self.tiles = None
        self.enemies = None
        self._should_update = True
        self.classMapping = ClassMapping()
    
    def draw_border(self, painter: QPainter, size: Tuple[float, float]) -> None:
        painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.green, Qt.NoBrush))
        painter.setRenderHint(QPainter.Antialiasing)
        points = [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]
        qpoints = [QPointF(point[0], point[1]) for point in points]
        polygon = QPolygonF(qpoints)
        painter.drawPolygon(polygon)

    def paintEvent(self, event):
        painter = QPainter(self)
        self.draw_border(painter, self.size)
        # if self.ram is not None:
        self.draw_tiles(painter)

        painter.end()


    def draw_tiles(self, painter: QPainter):
        if not self.tiles:
            return

        # Iterate only over existing tile locations stored in self.tiles
        for loc, class_id in self.tiles.items():
            col, row = loc  # loc is expected to be a tuple (row, col)

            painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
            painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
            x_start = 5 + (self.tile_width * col) + self.x_offset
            y_start = 5 + (self.tile_height * row)
            group_id = self.classMapping.get_group_id(class_id)
            if group_id:
                rgb = self.classMapping.get_color_by_group_id(group_id)
                color = QColor(*rgb)
                painter.setBrush(QBrush(color))
            else:
                # Handle other cases or use a default case if needed
                pass

            painter.drawRect(x_start, y_start, self.tile_width, self.tile_height)


    def draw_tiles_old(self, painter: QPainter):
        
        if not self.tiles:
            return
        for row in range(15):
            for col in range(16):
                loc = (row, col)
                

                if loc not in self.tiles:
                    continue


                tile = self.tiles[loc]
                painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                x_start = 5 + (self.tile_width * col) + self.x_offset
                y_start = 5 + (self.tile_height * row)
                

                if isinstance(tile, (StaticTileType, DynamicTileType, EnemyType, Item)):
                    # if tile.name == 'Empty' or tile.name == 'Fake':
                    #     continue
                    if ColorMap.has_name(tile.name):
                        rgb = ColorMap[tile.name].value
                        color = QColor(*rgb)
                    else:
                        # print(f"ColorMap has no color: {tile.name}")
                        rgb = (0,0,0)
                        color = QColor(*rgb)
                    painter.setBrush(QBrush(color))
                else:
                    pass
                painter.drawRect(x_start, y_start, self.tile_width, self.tile_height)

    def update_tiles(self, tiles):
        self.tiles = tiles
        self.update()

class GridWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.visualizer = GridVisualizer(self, size=(500, 300))
        self.setCentralWidget(self.visualizer)
        self.resize(800, 600)

    def update_tiles(self, tiles):
        # Example tiles update
        self.visualizer.update_tiles(tiles)



class GameFrameVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Pygame 설정
        pygame.init()
        self.x_pixel_num = 256
        self.y_pixel_num = 240
        self.game_screen = None
        # self.game_screen = pygame.display.set_mode((512, 480))

    def set_game_screen(self):
        self.game_screen = pygame.display.set_mode((self.x_pixel_num, self.y_pixel_num))


    def visualize(self, frame: pygame.surface):
        self.game_screen.blit(frame, (0, 0))
        pygame.display.flip()