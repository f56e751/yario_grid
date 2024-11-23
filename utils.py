# https://github.com/Chrispresso/SuperMarioBros-AI
# https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map 
# https://gist.github.com/1wErt3r/4048722
from collections import namedtuple
import numpy as np
from enum import Enum, unique
# TODO get_tile() 에서 지금은 StaticTileType만 반환하는데 DynamicTileType도 반환할 수 있도록 수정

class YoloLabel(Enum):
    Mario_small = 0
    Mario_big = 1
    Mario_fire = 2
    Enemy = 3


    Mushroom = 4
    Flower = 5
    Star = 6
    LifeUp = 7


    # Empty = 0x00
    Ground = 8
    Top_Pipe1 = 9
    Top_Pipe2 = 10
    Bottom_Pipe1 = 11
    Bottom_Pipe2 = 12
    Pipe_Horizontal = 13


    Flagpole_Top =  14
    Flagpole = 15
    Coin_Block = 16
    Coin_Block_End = 17
    Coin = 18

    Breakable_Block = 19

    @classmethod
    def get_value_from_name(cls, name):
        try:
            return cls[name].value
        except KeyError:
            return None  # 만약 이름이 없는 경우 None 반환

    
    

@unique
class EnemyType(Enum):
    Green_Koopa1 = 0x00
    Red_Koopa1   = 0x01
    Buzzy_Beetle = 0x02
    Red_Koopa2 = 0x03
    Green_Koopa2 = 0x04
    Hammer_Brother = 0x05
    Goomba      = 0x06
    Blooper = 0x07
    Bullet_Bill = 0x08
    Green_Koopa_Paratroopa = 0x09
    Grey_Cheep_Cheep = 0x0A
    Red_Cheep_Cheep = 0x0B
    Pobodoo = 0x0C
    Piranha_Plant = 0x0D
    Green_Paratroopa_Jump = 0x0E
    Bowser_Flame1 = 0x10
    Lakitu = 0x11
    Spiny_Egg = 0x12
    Fly_Cheep_Cheep = 0x14
    Bowser_Flame2 = 0x15
    PowerUp_Object = 0x2E 

    Generic_Enemy = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)
    
    @classmethod
    def get_enemy_name_from_value(cls, value: int) -> str:
        for enemy in cls:
            if enemy.value == value:
                return enemy
        return "Unknown Enemy"  # 이 값은 해당 코드에 맞는 적이 없을 경우 반환됩니다.

    @classmethod
    def has_name(cls, name) -> bool:
        return name in set(item.name for item in cls)

@unique
class StaticTileType(Enum):
    Empty = 0x00
    # Fake = 0x01
    Ground = 0x54
    Top_Pipe1 = 0x12
    Top_Pipe2 = 0x13
    Bottom_Pipe1 = 0x14
    Bottom_Pipe2 = 0x15
    Flagpole_Top =  0x24
    Flagpole = 0x25
    Coin_Block = 0xC0
    Coin_Block2 = 0xC1 
    Coin = 0xC2
    Breakable_Block = 0x51

    Pipe_Horizontal = 28 
    Coin_Block_End = 196


    Generic_Static_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)
    
    @classmethod
    def has_name(cls, name) -> bool:
        return name in set(item.name for item in cls)

@unique
class DynamicTileType(Enum):
    Mario = 0xAA

    Static_Lift1 = 0x24
    Static_Lift2 = 0x25
    Vertical_Lift1 = 0x26
    Vertical_Lift2 = 0x27
    Horizontal_Lift = 0x28
    Falling_Static_Lift = 0x29
    Horizontal_Moving_Lift=  0x2A
    Lift1 = 0x2B
    Lift2 = 0x2C
    Vine = 0x2F
    Flagpole = 0x30
    Start_Flag = 0x31
    Jump_Spring = 0x32
    Warpzone = 0x34
    Spring1 = 0x67
    Spring2 = 0x68

    Generic_Dynamic_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)
    

    
class Item(Enum):
    Mushroom = 0
    Flower = 1
    Star = 2
    LifeUp = 3

    @classmethod
    def get_item_name_from_value(cls, value: int) -> str:
        for item in cls:
            if item.value == value:
                return item
        return "Unknown Item"  # 이 값은 해당 코드에 맞는 적이 없을 경우 반환됩니다.
    
    @classmethod
    def has_name(cls, name) -> bool:
        return name in set(item.name for item in cls)




class ColorMap(Enum):
    Empty = (255, 255, 255)   # White
    Ground = (128, 43, 0)     # Brown
    Fake = (128, 43, 0)
    Mario = (0, 0, 255)
    Top_Pipe1 = (0, 15, 21)  # Dark Green
    Top_Pipe2 = (0, 15, 21)  # Dark Green
    Bottom_Pipe1 = (5, 179, 34)  # Light Green
    Bottom_Pipe2 = (5, 179, 34)  # Light Green
    Coin_Block1 = (219, 202, 18)  # Gold
    Coin_Block2 = (219, 202, 18)  # Gold
    Coin = (219, 202, 255)
    Breakable_Block = (79, 70, 25)  # Brownish

    Generic_Enemy = (255, 0, 20)  # Red
    # Generic_Static_Tile = (128, 43, 1)
    Generic_Static_Tile = (255, 0, 0) 
    Generic_Dynamic_Tile = (79, 70, 25)

    Flagpole_Top = (100,100,100)
    Flagpole = (100,0,100)


    Mushroom = (0,0,200)
    Flower = (0,200,0)
    Star = (200,0,0)
    LifeUp = (200,200,200)



    @classmethod
    def has_name(cls, name) -> bool:
        return name in set(item.name for item in cls)

Shape = namedtuple('Shape', ['width', 'height'])
Point = namedtuple('Point', ['x', 'y'])

class Tile(object):
    __slots__ = ['type']
    def __init__(self, type: Enum):
        self.type = type

class Enemy(object):
    def __init__(self, enemy_id: int, location: Point, tile_location: Point):
        enemy_type = EnemyType(enemy_id)
        self.type = EnemyType(enemy_id)
        self.location = location
        self.tile_location = tile_location

class RelativePositionCalculator():
    def __init__(self):
        self.pre_relative_position = 0
        self.count = 0

    def calculate_relative_position(self, mario_x, item_x):
        curr_relative_position = int(item_x) - int(mario_x)
        

        if self.count > 0 and self.pre_relative_position - curr_relative_position > 200:
            curr_relative_position += 256
        elif self.count > 0 and self.pre_relative_position - curr_relative_position < -200:
            curr_relative_position -= 256

        self.count += 1
        self.pre_relative_position = curr_relative_position
        return curr_relative_position
    
    def clear(self):
        self.pre_relative_position = 0
        self.count = 0



class SMB(object):
    # SMB can only load 5 enemies to the screen at a time.
    # Because of that we only need to check 5 enemy locations
    MAX_NUM_ENEMIES = 5
    PAGE_SIZE = 256
    NUM_BLOCKS = 8
    RESOLUTION = Shape(256, 240)
    NUM_TILES = 416  # 0x69f - 0x500 + 1
    NUM_SCREEN_PAGES = 2
    TOTAL_RAM = NUM_BLOCKS * PAGE_SIZE

    sprite = Shape(width=16, height=16)
    resolution = Shape(256, 240)
    status_bar = Shape(width=resolution.width, height=2*sprite.height)

    xbins = list(range(16, resolution.width, 16))
    ybins = list(range(16, resolution.height, 16))

    previous_item_state = False 
    relative_pos_calculator = RelativePositionCalculator()

    tiles = {}

    @unique
    class RAMLocations(Enum):
        # Since the max number of enemies on the screen is 5, the addresses for enemies are
        # the starting address and span a total of 5 bytes. This means Enemy_Drawn + 0 is the
        # whether or not enemy 0 is drawn, Enemy_Drawn + 1 is enemy 1, etc. etc.
        Enemy_Drawn = 0x0F
        Enemy_Type = 0x16
        Enemy_X_Position_In_Level = 0x6E
        Enemy_X_Position_On_Screen = 0x87
        Enemy_Y_Position_On_Screen = 0xCF

        Player_X_Postion_In_Level       = 0x06D
        Player_X_Position_On_Screen     = 0x086

        Player_X_Position_Screen_Offset = 0x3AD
        Player_Y_Position_Screen_Offset = 0x3B8
        Enemy_X_Position_Screen_Offset = 0x3AE

        Player_Y_Pos_On_Screen = 0xCE
        Player_Vertical_Screen_Position = 0xB5


        Player_State = 0x000E
        Mario_Powerup_State = 0x0756 # 0: small, 1: big,  => 2 : firey
        Coin_Num = 0x075E
        Lives = 0x075A
        World = 0x075F
        Level = 0x0760

        PowerUp_On_Screen = 0x001B # 0x00 - No, 0x2E - Yes
        PowerUp_Type = 0x0039 # 0 - Mushroom, 1 - Flower, 2 - Star, 3 - 1up
        PowerUp_X_Position_On_Screen = 0x008C
        PowerUp_X_Position_Screen_Offset = 0x03B3
        PowerUp_Sprite_Y = 0x00D4
        PowerUp_Vertical_Screen_Pos = 0x00BB


        

    def get_ram_info(cls, ram, rAMLocations: RAMLocations):
        return ram[rAMLocations.value]

    @classmethod
    def get_item_pos(cls, ram):
        mario_level = cls.get_mario_location_in_level(ram)
        mario_screen = cls.get_mario_location_on_screen(ram)

        x_start = mario_level.x - mario_screen.x

        if not cls.is_item_on_screen(ram):
            cls.previous_item_state = False
            return None
        else:
            if cls.previous_item_state == False: # 아이템 최초 생성 시점
                cls.relative_pos_calculator.clear()
                cls.previous_item_state = True


        PowerUp_Sprite_Y = ram[cls.RAMLocations.PowerUp_Sprite_Y.value]
        

        PowerUp_X_Position_On_Screen = ram[cls.RAMLocations.PowerUp_X_Position_On_Screen.value]
        vertical_screen_pos = ram[cls.RAMLocations.PowerUp_Vertical_Screen_Pos.value]
        item_y_pos = PowerUp_Sprite_Y * vertical_screen_pos
        
        # print(f"mario: ({mario_x}, {mario_y})")

        mario_x = ram[cls.RAMLocations.Player_X_Position_On_Screen.value]
        mario_x_offset = ram[cls.RAMLocations.Player_X_Position_Screen_Offset.value]

        relative_position = cls.relative_pos_calculator.calculate_relative_position(mario_x, PowerUp_X_Position_On_Screen)
        # print(Point(mario_x_offset + relative_position, item_y_pos), mario_screen)
        return Point(mario_x_offset + relative_position, item_y_pos)

    @classmethod
    def get_item_row_col(cls, ram):
        item_pos = cls.get_item_pos(ram)

        if item_pos != None:
            x = item_pos.x
            y = item_pos.y
            x += 12
            y += 16
            col = x // 16
            row = (y - 0) // 16
            # cls.get_item_type(ram)

            return(row,col)
        else:
            return None


    @classmethod
    def get_x_start(cls, ram):
        mario_level = cls.get_mario_location_in_level(ram)
        mario_screen = cls.get_mario_location_on_screen(ram)

        x_start = mario_level.x - mario_screen.x
        return x_start
    
    @classmethod
    def is_position_in_range(cls, ram, position):
        x_start = cls.get_x_start(ram)
        x_end = x_start + 256
        if x_start <= position and position <= x_end:
            return True
        else:
            return False


    @classmethod
    def get_item_type(cls, ram): 
        item_type = ram[cls.RAMLocations.PowerUp_Type.value]
        # print(f"item type: {item_type}")
        return item_type # 0 - Mushroom, 1 - Flower, 2 - Star, 3 - 1up

    @classmethod
    def is_item_on_screen(cls, ram):
        powerUp_On_Screen = ram[cls.RAMLocations.PowerUp_On_Screen.value]
        if powerUp_On_Screen == 0x00:
            return False
        elif powerUp_On_Screen == 0x2E:
            return True
        else:
            return False
            
    @classmethod
    def get_enemy_locations(cls, ram: np.ndarray):
        # We only care about enemies that are drawn. Others may?? exist
        # in memory, but if they aren't on the screen, they can't hurt us.
        # enemies = [None for _ in range(cls.MAX_NUM_ENEMIES)]
        
        # enemies = []
        enemies = {} # {Enemy(0x6, location, tile_location): enemy_id}

        for enemy_num in range(cls.MAX_NUM_ENEMIES):
            enemy = ram[cls.RAMLocations.Enemy_Drawn.value + enemy_num]
            # Is there an enemy? 1/0
            if enemy:
                # Get the enemy X location.
                x_pos_level = ram[cls.RAMLocations.Enemy_X_Position_In_Level.value + enemy_num]
                x_pos_screen = ram[cls.RAMLocations.Enemy_X_Position_On_Screen.value + enemy_num]
                enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen #- ram[0x71c]
                # print(ram[0x71c])
                # enemy_loc_x = ram[cls.RAMLocations.Enemy_X_Position_Screen_Offset.value + enemy_num]
                # Get the enemy Y location.
                enemy_loc_y = ram[cls.RAMLocations.Enemy_Y_Position_On_Screen.value + enemy_num]
                # Set location
                location = Point(enemy_loc_x, enemy_loc_y)
                ybin = np.digitize(enemy_loc_y, cls.ybins)
                xbin = np.digitize(enemy_loc_x, cls.xbins)
                tile_location = Point(xbin, ybin)

                # Grab the id
                enemy_id = ram[cls.RAMLocations.Enemy_Type.value + enemy_num]
                # Create enemy-
                e = Enemy(0x6, location, tile_location)

                # enemies.append(e)
                enemies[e] = enemy_id

        return enemies

    @classmethod
    def get_mario_location_in_level(cls, ram: np.ndarray) -> Point:
        mario_x = ram[cls.RAMLocations.Player_X_Postion_In_Level.value] * 256 + ram[cls.RAMLocations.Player_X_Position_On_Screen.value]
        mario_y = ram[cls.RAMLocations.Player_Y_Position_Screen_Offset.value]
        return Point(mario_x, mario_y)

    @classmethod
    def get_mario_score(cls, ram: np.ndarray) -> int:
        multipllier = 10
        score = 0
        for loc in range(0x07DC, 0x07D7-1, -1):
            score += ram[loc]*multipllier
            multipllier *= 10

        return score

    @classmethod
    def get_mario_location_on_screen(cls, ram: np.ndarray):
        mario_x = ram[cls.RAMLocations.Player_X_Position_Screen_Offset.value]
        mario_y = ram[cls.RAMLocations.Player_Y_Pos_On_Screen.value] * ram[cls.RAMLocations.Player_Vertical_Screen_Position.value] + cls.sprite.height
        return Point(mario_x, mario_y)

    @classmethod
    def get_tile_type(cls, ram:np.ndarray, delta_x: int, delta_y: int, mario: Point):
        x = mario.x + delta_x
        y = mario.y + delta_y + cls.sprite.height

        # Tile locations have two pages. Determine which page we are in
        page = (x // 256) % 2
        # Figure out where in the page we are
        sub_page_x = (x % 256) // 16
        sub_page_y = (y - 32) // 16  # The PPU is not part of the world, coins, etc (status bar at top)
        if sub_page_y not in range(13):# or sub_page_x not in range(16):
            return StaticTileType.Empty.value

        addr = 0x500 + page*208 + sub_page_y*16 + sub_page_x
        return ram[addr]

    @classmethod
    def get_tile_loc(cls, x, y):
        row = np.digitize(y, cls.ybins) - 2
        col = np.digitize(x, cls.xbins)
        return (row, col)
    
    @classmethod
    def get_start_x(cls, ram):
        mario_level = cls.get_mario_location_in_level(ram)
        mario_screen = cls.get_mario_location_on_screen(ram)

        x_start = mario_level.x - mario_screen.x
        return x_start
    
    @classmethod
    def groundTest(cls, ram):
        x_pos = cls.get_x_start(ram)
        isGround = False
        for y_pos in range(240):
            tile_type = cls.get_tile(x_pos, y_pos, ram)
            if (tile_type == StaticTileType.Ground):
                if isGround == False:
                    print(y_pos)
                    isGround = True

    @classmethod
    def itemBoxTest(cls, ram):
        is_search = False
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        start_x = cls.get_start_x(ram)
        # Coin_Block의 좌표를 탐색
        for item, value in cls.tiles.items():
            if value == StaticTileType.Coin_Block:
                is_search = True
                coin_x = item[1] * 16 + start_x
                coin_y = item[0] * 16

                # print(coin_x, coin_y)



                # Coin_Block이 있는 근방 10 픽셀 이내를 탐색하여 범위 설정
                for y in range(coin_y - 20, coin_y + 21):
                    for x in range(coin_x - 20, coin_x + 21):
                        tile_type = cls.get_tile(x, y, ram)
                        if tile_type == StaticTileType.Coin_Block:
                            min_x = min(min_x, x)
                            max_x = max(max_x, x)
                            min_y = min(min_y, y)
                            max_y = max(max_y, y)

        if is_search:
            print(f"x: ({min_x- start_x}, {max_x - start_x}), y: ({min_y}, {max_y})")
        else:
            print("No Coin_Block found.")




    @classmethod
    def get_tiles(cls, ram: np.ndarray, detailed_enemies=True):
        cls.tiles = {}
        row = 0
        col = 0

        mario_level = cls.get_mario_location_in_level(ram)
        mario_screen = cls.get_mario_location_on_screen(ram)

        x_start = mario_level.x - mario_screen.x

        enemies = cls.get_enemy_locations(ram)
        y_start = 0
        # mx, my = cls.get_mario_location_in_level(ram)
        # my += 16
        # Set mx to be within the screen offset
        mx = ram[cls.RAMLocations.Player_X_Position_Screen_Offset.value]


        for y_pos in range(y_start, 240, 16):
            for x_pos in range(x_start, x_start + 256, 16):
                loc = (row, col)
                tile_type = cls.get_tile(x_pos, y_pos, ram)

                # Assign tile type directly using the Enum returned by get_tile
                cls.tiles[loc] = tile_type

                for enemy in enemies:
                    ex = enemy.location.x
                    ey = enemy.location.y + 8
                    # Since we can only discriminate within 8 pixels, if it falls within this bound, count it as there
                    if abs(x_pos - ex) <= 8 and abs(y_pos - ey) <= 8:
                        if detailed_enemies:
                            id = enemies[enemy]
                            cls.tiles[loc] = EnemyType.get_enemy_name_from_value(id)
                        else:
                            cls.tiles[loc] = EnemyType.Generic_Enemy
                # Next col
                col += 1
            # Move to next row
            col = 0
            row += 1

        mario_locs = cls.get_mario_row_col(ram)
        for mario_loc in mario_locs:
            cls.tiles[mario_loc] = DynamicTileType.Mario
        # loc = (mario_row, mario_col)
        
        item_loc = cls.get_item_row_col(ram)
        if item_loc != None:
            item_type = cls.get_item_type(ram)
            cls.tiles[item_loc] = Item.get_item_name_from_value(item_type)
        # player_state = ram[cls.RAMLocations.Player_State.value]
        return cls.tiles
    

    @classmethod
    def get_yolo_format(cls, ram):
        yolo_format = {} # {object_name: [[x,y], [x_length, y_length]], [[x,y], [x_length, y_length]] } 
        def make_key_for_dict(dict: dict, key):
            if key not in dict:
                dict[key] = []
        
        x_unit_length = 1/16
        y_unit_length = 1/15

        x_start = cls.get_x_start(ram)
        res = x_start % 16
        if res <= 8:
            x_start_adjust = x_start + (16 - res) # x_start_adjust는 잘리지 않고 블록 중간에서 시작하는 x 좌표
        else:
            x_start_adjust = x_start + (32 - res)
        
        y_start_adjust = 8 
        y_iter_num = 14

        if res == 8:
            x_iter_num = 15
        else:
            x_iter_num = 14

        x_start_adjust += 1
        for i in range(y_iter_num):
            for j in range(x_iter_num):
                x = x_start_adjust + j * 17
                y = y_start_adjust + i * 16


                tile = cls.get_tile(x, y, ram)
                # tile_name = tile.name
                try:
                    tile_name = tile.name
                except AttributeError:
                    print("Error accessing tile.name. Tile is:", tile)
                    tile_name = "Unknown"  # 에러가 발생했을 때 기본값 설정 (필요 시)

                if tile_name == "Empty": # empty는 학습 X
                    continue

                make_key_for_dict(yolo_format, tile_name)
                x_on_screen = x - x_start
                y_on_screen = y

                x_yolo = x_on_screen / 256
                y_yolo = (y_on_screen + 8) / 240 # 확인해보니 위에 8픽셀은 빈 공간임
                yolo_format[tile_name].append([[x_yolo,y_yolo], [x_unit_length, y_unit_length]])

        mario_x, mario_y = cls.get_mario_location_on_screen(ram)
        mario_x = (mario_x + 4) / 256
        mario_y = (mario_y + 16) / 240
        mario_state = cls.get_mario_state(ram) # 0: small, 1: big,  => 2 : firey
        
        if mario_state == 0:
            make_key_for_dict(yolo_format, "Mario_small")
            yolo_format["Mario_small"].append([[mario_x, mario_y],[x_unit_length, y_unit_length]])
        elif mario_state == 1:
            make_key_for_dict(yolo_format, "Mario_big")
            yolo_format["Mario_big"].append([[mario_x, mario_y - 8 / 240],[x_unit_length, y_unit_length * 2]])
        else:
            make_key_for_dict(yolo_format, "Mario_fire")
            yolo_format["Mario_fire"].append([[mario_x, mario_y - 8 / 240],[x_unit_length, y_unit_length * 2]])

        

        # print(mario_x, mario_y)
        enemies = cls.get_enemy_locations(ram)
        for enemy in enemies:
            make_key_for_dict(yolo_format, "Enemy")
            ex = enemy.location.x - x_start + 8
            ey = enemy.location.y + 8 + 16
            x_yolo = ex / 256
            y_yolo = ey / 240
            if x_yolo > 0.98 or x_yolo < 0.02:
                continue
            yolo_format["Enemy"].append([[x_yolo, y_yolo],[x_unit_length, y_unit_length]])

        # if 'Enemy' in yolo_format:
        #     print(f"enemy num is {len(yolo_format['Enemy'])}")

        item = cls.get_item_pos(ram)
        if item != None:
            x_yolo = item.x / 256
            y_yolo = item.y / 240

            item_type = cls.get_item_type(ram) # 0 - Mushroom, 1 - Flower, 2 - Star, 3 - 1up
            item_name = Item.get_item_name_from_value(item_type).name # Mushroom, Flower, Star, LifeUp
            make_key_for_dict(yolo_format, item_name)
            yolo_format[item_name].append([[x_yolo, y_yolo],[x_unit_length, y_unit_length]])



        for key in list(yolo_format.keys()):  # 원래 키를 수정해야 하므로 list로 복사해서 사용
            label_value = YoloLabel.get_value_from_name(key)
            if label_value is not None:
                yolo_format[label_value] = yolo_format.pop(key)  # 기존 키를 Enum 값으로 변경
            else:
                raise KeyError(f"The key '{key}' does not exist in YoloLabel Enum.")



        return yolo_format



    @classmethod
    def get_yolo_format_new(cls, ram):
        yolo_format = {} # {object_name: [[x,y], [x_length, y_length]], [[x,y], [x_length, y_length]] } 
        def make_key_for_dict(dict: dict, key):
            if key not in dict:
                dict[key] = []
        
        x_unit_length = 1/16
        y_unit_length = 1/15

        x_start = cls.get_x_start(ram)
        res = x_start % 16
        if res <= 8:
            x_start_adjust = x_start + (16 - res) # x_start_adjust는 잘리지 않고 블록 중간에서 시작하는 x 좌표
        else:
            x_start_adjust = x_start + (32 - res)
        
        y_start_adjust = 18
        y_iter_num = 13

        if res == 8:
            x_iter_num = 15
        else:
            x_iter_num = 14

        x_start_adjust += 2
        for i in range(y_iter_num):
            for j in range(x_iter_num):
                x = x_start_adjust + j * 17
                y = y_start_adjust + i * 17


                tile = cls.get_tile(x, y, ram)
                # tile_name = tile.name
                try:
                    tile_name = tile.name
                except AttributeError:
                    print("Error accessing tile.name. Tile is:", tile)
                    tile_name = "Unknown"  # 에러가 발생했을 때 기본값 설정 (필요 시)

                if tile_name == "Empty": # empty는 학습 X
                    continue

                make_key_for_dict(yolo_format, tile_name)
                x_on_screen = x - x_start
                y_on_screen = y

                x_yolo = x_on_screen / 256
                y_yolo = (y_on_screen) / 240 # 확인해보니 위에 8픽셀은 빈 공간임
                yolo_format[tile_name].append([[x_yolo,y_yolo], [x_unit_length, y_unit_length]])

        mario_x, mario_y = cls.get_mario_location_on_screen(ram)
        mario_x = (mario_x + 4) / 256
        mario_y = (mario_y + 16) / 240
        mario_state = cls.get_mario_state(ram) # 0: small, 1: big,  => 2 : firey
        
        if mario_state == 0:
            make_key_for_dict(yolo_format, "Mario_small")
            yolo_format["Mario_small"].append([[mario_x, mario_y],[x_unit_length, y_unit_length]])
        elif mario_state == 1:
            make_key_for_dict(yolo_format, "Mario_big")
            yolo_format["Mario_big"].append([[mario_x, mario_y - 8 / 240],[x_unit_length, y_unit_length * 2]])
        else:
            make_key_for_dict(yolo_format, "Mario_fire")
            yolo_format["Mario_fire"].append([[mario_x, mario_y - 8 / 240],[x_unit_length, y_unit_length * 2]])

        

        # print(mario_x, mario_y)
        enemies = cls.get_enemy_locations(ram)
        for enemy in enemies:
            make_key_for_dict(yolo_format, "Enemy")
            ex = enemy.location.x - x_start + 8
            ey = enemy.location.y + 8 + 16
            x_yolo = ex / 256
            y_yolo = ey / 240
            if x_yolo > 0.98 or x_yolo < 0.02:
                continue
            yolo_format["Enemy"].append([[x_yolo, y_yolo],[x_unit_length * 1.2, y_unit_length * 1.2]])

        # if 'Enemy' in yolo_format:
        #     print(f"enemy num is {len(yolo_format['Enemy'])}")

        item = cls.get_item_pos(ram)
        if item != None:
            x_yolo = item.x / 256
            y_yolo = (item.y + 16) / 240

            item_type = cls.get_item_type(ram) # 0 - Mushroom, 1 - Flower, 2 - Star, 3 - 1up
            item_name = Item.get_item_name_from_value(item_type).name # Mushroom, Flower, Star, LifeUp
            make_key_for_dict(yolo_format, item_name)
            yolo_format[item_name].append([[x_yolo, y_yolo],[x_unit_length, y_unit_length]])



        for key in list(yolo_format.keys()):  # 원래 키를 수정해야 하므로 list로 복사해서 사용
            label_value = YoloLabel.get_value_from_name(key)
            if label_value is not None:
                yolo_format[label_value] = yolo_format.pop(key)  # 기존 키를 Enum 값으로 변경
            else:
                raise KeyError(f"The key '{key}' does not exist in YoloLabel Enum.")



        return yolo_format
    

    @classmethod
    def get_yolo_format_for_game(cls, ram):
        # yolo train을 위한 위치조정 제거 
        yolo_format = {} # {object_name: [[x,y], [x_length, y_length]], [[x,y], [x_length, y_length]] } 
        def make_key_for_dict(dict: dict, key):
            if key not in dict:
                dict[key] = []
        
        x_unit_length = 1/16
        y_unit_length = 1/15

        x_start = cls.get_x_start(ram)
        res = x_start % 16
        if res <= 8:
            x_start_adjust = x_start + (16 - res) # x_start_adjust는 잘리지 않고 블록 중간에서 시작하는 x 좌표
        else:
            x_start_adjust = x_start + (32 - res)
        
        y_start_adjust = 18
        y_iter_num = 13

        if res == 8:
            x_iter_num = 15
        else:
            x_iter_num = 14

        x_start_adjust += 2
        for i in range(y_iter_num):
            for j in range(x_iter_num):
                # x = x_start_adjust + j * 17
                # y = y_start_adjust + i * 17

                x = x_start_adjust + j * 16
                y = y_start_adjust + i * 16


                tile = cls.get_tile(x, y, ram)
                # tile_name = tile.name
                try:
                    tile_name = tile.name
                except AttributeError:
                    print("Error accessing tile.name. Tile is:", tile)
                    tile_name = "Unknown"  # 에러가 발생했을 때 기본값 설정 (필요 시)

                if tile_name == "Empty": # empty는 학습 X
                    continue

                make_key_for_dict(yolo_format, tile_name)
                x_on_screen = x - x_start
                y_on_screen = y

                x_yolo = x_on_screen / 256
                y_yolo = (y_on_screen) / 240 # 확인해보니 위에 8픽셀은 빈 공간임
                yolo_format[tile_name].append([[x_yolo,y_yolo], [x_unit_length, y_unit_length]])

        mario_x, mario_y = cls.get_mario_location_on_screen(ram)
        mario_x = (mario_x) / 256
        mario_y = (mario_y + 8) / 240
        # mario_y = (mario_y + 16) / 240
        mario_state = cls.get_mario_state(ram) # 0: small, 1: big,  => 2 : firey
        
        if mario_state == 0:
            make_key_for_dict(yolo_format, "Mario_small")
            yolo_format["Mario_small"].append([[mario_x, mario_y],[x_unit_length, y_unit_length]])
        elif mario_state == 1:
            make_key_for_dict(yolo_format, "Mario_big")
            yolo_format["Mario_big"].append([[mario_x, mario_y - 8 / 240],[x_unit_length, y_unit_length * 2]])
        else:
            make_key_for_dict(yolo_format, "Mario_fire")
            yolo_format["Mario_fire"].append([[mario_x, mario_y - 8 / 240],[x_unit_length, y_unit_length * 2]])

        

        # print(mario_x, mario_y)
        enemies = cls.get_enemy_locations(ram)
        for enemy in enemies:
            make_key_for_dict(yolo_format, "Enemy")
            # ex = enemy.location.x - x_start + 8
            ex = enemy.location.x - x_start
            ey = enemy.location.y + 8
            x_yolo = ex / 256
            y_yolo = ey / 240
            if x_yolo > 0.98 or x_yolo < 0.02:
                continue
            yolo_format["Enemy"].append([[x_yolo, y_yolo],[x_unit_length * 1.2, y_unit_length * 1.2]])

        # if 'Enemy' in yolo_format:
        #     print(f"enemy num is {len(yolo_format['Enemy'])}")

        item = cls.get_item_pos(ram)
        if item != None:
            x_yolo = item.x / 256
            y_yolo = (item.y + 16) / 240
            # y_yolo = (item.y) / 240

            item_type = cls.get_item_type(ram) # 0 - Mushroom, 1 - Flower, 2 - Star, 3 - 1up
            item_name = Item.get_item_name_from_value(item_type).name # Mushroom, Flower, Star, LifeUp
            make_key_for_dict(yolo_format, item_name)
            yolo_format[item_name].append([[x_yolo, y_yolo],[x_unit_length, y_unit_length]])



        for key in list(yolo_format.keys()):  # 원래 키를 수정해야 하므로 list로 복사해서 사용
            label_value = YoloLabel.get_value_from_name(key)
            if label_value is not None:
                yolo_format[label_value] = yolo_format.pop(key)  # 기존 키를 Enum 값으로 변경
            else:
                raise KeyError(f"The key '{key}' does not exist in YoloLabel Enum.")



        return yolo_format
    

        

    @classmethod
    def get_yolo_format_unit_test(cls, ram):
        yolo_format = {} # {object_name: [[x,y], [x_length, y_length]], [[x,y], [x_length, y_length]] } 
        def make_key_for_dict(dict: dict, key):
            if key not in dict:
                dict[key] = []
        
        x_unit_length = 1/16
        y_unit_length = 1/15

        x_start = cls.get_x_start(ram)
        res = x_start % 16
        if res <= 8:
            x_start_adjust = x_start + (16 - res) # x_start_adjust는 잘리지 않고 블록 중간에서 시작하는 x 좌표
        else:
            x_start_adjust = x_start + (32 - res)
        
        y_start_adjust = 18
        y_iter_num = 13

        if res == 8:
            x_iter_num = 15
        else:
            x_iter_num = 14

        x_start_adjust += 1
        for i in range(y_iter_num):
            for j in range(x_iter_num):
                x = x_start_adjust + j * 17
                y = y_start_adjust + i * 17


                tile_name = "Ground"
                make_key_for_dict(yolo_format, tile_name)
                x_on_screen = x - x_start
                y_on_screen = y

                x_yolo = x_on_screen / 256
                y_yolo = (y_on_screen) / 240 # 확인해보니 위에 8픽셀은 빈 공간임
                yolo_format[tile_name].append([[x_yolo,y_yolo], [x_unit_length, y_unit_length]])



        for key in list(yolo_format.keys()):  # 원래 키를 수정해야 하므로 list로 복사해서 사용
            label_value = YoloLabel.get_value_from_name(key)
            if label_value is not None:
                yolo_format[label_value] = yolo_format.pop(key)  # 기존 키를 Enum 값으로 변경
            else:
                raise KeyError(f"The key '{key}' does not exist in YoloLabel Enum.")



        return yolo_format


    @classmethod
    def is_recordable(cls, ram):
        # Player's state
        # 0x00 - Leftmost of screen
        # 0x01 - Climbing vine
        # 0x02 - Entering reversed-L pipe
        # 0x03 - Going down a pipe
        # 0x04 - Autowalk
        # 0x05 - Autowalk
        # 0x06 - Player dies
        # 0x07 - Entering area
        # 0x08 - Normal
        # 0x09 - Transforming from Small to Large (cannot move)
        # 0x0A - Transforming from Large to Small (cannot move)
        # 0x0B - Dying
        # 0x0C - Transforming to Fire Mario (cannot move)
        player_state = ram[cls.RAMLocations.Player_State.value]
        # if player_state != 8:
        #     print(f"player_state: {player_state}")

        if player_state in [0, 0x0B, 0x06, 0x07, 0x03, 0x05, 0x02, 0x04]:
            # 로딩중, 죽을때 제외함
            return False

        return True
    
    @classmethod
    def is_world_cleared(cls, ram):
        player_state = ram[cls.RAMLocations.Player_State.value]
        if player_state == 4:
            return True
        else:
            return False

    
    @classmethod
    def is_dead(cls, ram):
        player_state = ram[cls.RAMLocations.Player_State.value]
        # print(f"player state: {player_state}")
        mario_x, mario_y = cls.get_mario_location_on_screen(ram)
        
        if mario_y > 256:
            # print(f"mario_y: {mario_y}")
            return True
        if player_state == 0x0B:
            return True
        else:
            return False

    
    @classmethod
    def get_mario_state(cls, ram):
        # 0: small, 1: big,  => 2 : firey
        mario_state = ram[cls.RAMLocations.Mario_Powerup_State.value]
        # print(f"mario_state: {mario_state}")
        return mario_state
    
    @classmethod
    def get_Coins(cls, ram):
        coins = ram[cls.RAMLocations.Coin_Num.value]
        # print(f"coins: {coins}")
        return coins
    
    @classmethod
    # 1 낮게 나옴 (ex 1-1 -> 0-0)
    def get_World(cls, ram):
        world = ram[cls.RAMLocations.World.value]
        # print(f"world: {world}")
        return world

    @classmethod
    # 1 낮게 나옴 (ex 1-1 -> 0-0)
    def get_Level(cls,ram):
        level = ram[cls.RAMLocations.Level.value]
        # print(f"level: {level}")
        return level
    
    @classmethod
    def get_Lives(cls,ram):
        lives = ram[cls.RAMLocations.Lives.value]
        # print(f"lives: {lives}")
        return lives
    

    @classmethod
    def get_Time(cls, ram: np.ndarray) -> int:
        multipllier = 1
        time = 0
        for loc in range(0x07FA, 0x07F8-1, -1):
            time += ram[loc]*multipllier
            multipllier *= 10

        # print(time)
        return time

    @classmethod
    def get_mario_row_col(cls, ram):
        x, y = cls.get_mario_location_on_screen(ram)
        # Adjust 16 for PPU
        y = ram[cls.RAMLocations.Player_Y_Position_Screen_Offset.value] + 16
        x += 12
        col = x // 16
        row = (y - 0) // 16
        mario_powerup_state = cls.get_mario_state(ram)
        if mario_powerup_state == 0:
            return [(row, col)]
        else:
            return [(row, col), (row - 1, col)]


    @classmethod
    def get_tile(cls, x, y, ram):
        page = (x // 256) % 2
        sub_x = (x % 256) // 16
        sub_y = (y - 32) // 16  # The PPU is not part of the world (status bar at top)

        # 이 부분 없으면 화면 위에 타일이 있다고 뜸
        if sub_y not in range(13):
            return StaticTileType.Empty

        addr = 0x500 + page * 208 + sub_y * 16 + sub_x
        tile_code = ram[addr]

        # Top_Pipe1 = 0x12 
        # Top_Pipe2 = 0x13
        # 아래로 내려갈 수 있는 파이프는 16, 17로 뜸
        if tile_code == 16:
            tile_code = 0x12
        if tile_code == 17:
            tile_code = 0x13

        # Empty = 0x00
        # 공중에 있는 투명한 아이템 블록이 96으로 뜸
        if tile_code == 96:
            tile_code = 0x00

        # Breakable_Block = 0x51
        # 지하의 파란 블록이 82로 뜸
        if tile_code == 82:
            tile_code = 0x51

        # Pipe_Horizontal = 28 
        # 수평 파이프가 28 ~ 33으로 뜸
        if tile_code in range(28,34):
            tile_code = 28

        # Breakable_Block = 0x51
        # 여러 번 연타 가능한 블록이 87, 88로 뜸
        if tile_code in [87, 88, 90, 93, 94]:
            tile_code = 0x51

        # Ground = 0x54
        # 계단이 97로 뜸
        if tile_code == 97:
            tile_code = 0x54

        # Coin_Block_End = 196
        # 코인 블록을 칠 때, 35로 뜸
        if tile_code == 35:
            tile_code = 196

        # Coin_Block1 = 0xC0
        # Coin_Block2 = 0xC1 
        # 코인 블록 하나로 취급
        if tile_code == 0xC1:
            tile_code = 0xC0


        try:
            return StaticTileType(tile_code)
        except ValueError:
            # If the code is not in StaticTileType, return a generic static tile type
            print("StaticTileType.Generic_Static_Tile")
            print(tile_code)
            return StaticTileType.Generic_Static_Tile
        

    @classmethod
    def get_tile_dynamic(cls, x, y, ram):
        page = (x // 256) % 2
        sub_x = (x % 256) // 16
        sub_y = (y - 32) // 16  # The PPU is not part of the world (status bar at top)

        addr = 0x5d0 + page * 208 + sub_y * 16 + sub_x
        tile_code = ram[addr]

        try:
            return DynamicTileType(tile_code)
        except ValueError:
            # If the code is not in StaticTileType, return a generic static tile type
            return None
        
