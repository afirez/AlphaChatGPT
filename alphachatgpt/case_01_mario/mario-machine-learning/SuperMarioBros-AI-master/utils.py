from collections import namedtuple
import numpy as np
from enum import Enum, unique

#定义敌人常量
#装饰器，使枚举类型不单成员名称不得一样，而且成员的值也不得一样
@unique
#继承枚举类型
class EnemyType(Enum):
    Green_Koopa1 = 0x00 #那个乌龟
    Red_Koopa1   = 0x01
    Buzzy_Beetle = 0x02 #盔甲虫
    Red_Koopa2 = 0x03
    Green_Koopa2 = 0x04
    Hammer_Brother = 0x05
    Goomba      = 0x06
    Blooper = 0x07
    Bullet_Bill = 0x08
    Green_Koopa_Paratroopa = 0x09
    Grey_Cheep_Cheep = 0x0A
    Red_Cheep_Cheep = 0x0B # 红色的鱼
    Pobodoo = 0x0C
    Piranha_Plant = 0x0D
    Green_Paratroopa_Jump = 0x0E
    Bowser_Flame1 = 0x10
    Lakitu = 0x11
    Spiny_Egg = 0x12
    Fly_Cheep_Cheep = 0x14
    Bowser_Flame2 = 0x15

    Generic_Enemy = 0xFF

    #对应的函数不需要实例化，不需要 self 参数
    #但第一个参数需要是用来表示自身类的cls参数
    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

#静态方块常量
@unique
class StaticTileType(Enum):
    Empty = 0x00
    Fake = 0x01
    Ground = 0x54
    Top_Pipe1 = 0x12
    Top_Pipe2 = 0x13
    Bottom_Pipe1 = 0x14
    Bottom_Pipe2 = 0x15
    Flagpole_Top =  0x24
    Flagpole = 0x25
    Coin_Block1 = 0xC0
    Coin_Block2 = 0xC1 
    Coin = 0xC2
    Breakable_Block = 0x51

    Generic_Static_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

#动态方块常量
@unique
class DynamicTileType(Enum):
    Mario = 0xAA            #马里奥

    Static_Lift1 = 0x24     #静止提升1
    Static_Lift2 = 0x25     #静止提升2
    Vertical_Lift1 = 0x26   #竖直提升1
    Vertical_Lift2 = 0x27   #竖直提升2
    Horizontal_Lift = 0x28  #水平提升
    Falling_Static_Lift = 0x29     #下坠提升
    Horizontal_Moving_Lift=  0x2A  #水平移动提升
    Lift1 = 0x2B
    Lift2 = 0x2C  
    Vine = 0x2F      #藤蔓
    Flagpole = 0x30  #旗杆
    Start_Flag = 0x31   #开始标志
    Jump_Spring = 0x32  #跳跃弹簧
    Warpzone = 0x34
    Spring1 = 0x67  #弹簧1
    Spring2 = 0x68  #弹簧2

    Generic_Dynamic_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)

#定义方块的绘制颜色，颜色对应
class ColorMap(Enum):
    Empty = (255, 255, 255)   # White
    Ground = (128, 43, 0)     # Brown
    Fake = (128, 43, 0)
    Mario = (0, 0, 255)
    Goomba = (255, 0, 20)
    Top_Pipe1 = (0, 15, 21)  # Dark Green
    Top_Pipe2 = (0, 15, 21)  # Dark Green
    Bottom_Pipe1 = (5, 179, 34)  # Light Green
    Bottom_Pipe2 = (5, 179, 34)  # Light Green
    Coin_Block1 = (219, 202, 18)  # Gold
    Coin_Block2 = (219, 202, 18)  # Gold
    Breakable_Block = (79, 70, 25)  # Brownish

    Generic_Enemy = (255, 0, 20)  # Red
    Generic_Static_Tile = (128, 43, 0) 
    Generic_Dynamic_Tile = (79, 70, 25)

Shape = namedtuple('Shape', ['width', 'height'])
Point = namedtuple('Point', ['x', 'y'])

#方块类
class Tile(object):
    __slots__ = ['type']
    def __init__(self, type: Enum):
        self.type = type

#敌人类
class Enemy(object):
    def __init__(self, enemy_id: int, location: Point, tile_location: Point):
        enemy_type = EnemyType(enemy_id)
        self.type = EnemyType(enemy_id)
        self.location = location
        self.tile_location = tile_location



#敌人位置加载器？
class SMB(object):
    # SMB 在屏幕上一次只能加载5个敌人
    # 因为我们只需要检查5个敌人的位置
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


    #敌人和玩家的绘制位置常量
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

    @classmethod
    def get_enemy_locations(cls, ram: np.ndarray):
        # 我们只关心画出来的敌人。其它的呢？从内存里排除掉。而且他们不在屏幕上当然也不会伤到我们。
        # enemies = [None for _ in range(cls.MAX_NUM_ENEMIES)]
        enemies = []

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

                enemies.append(e)

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

    #获得釉面类型
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

    # 从虚拟机内充中读取釉面信息
    @classmethod
    def get_tiles(cls, ram: np.ndarray):
        tiles = {}

        mario_level = cls.get_mario_location_in_level(ram)
        mario_screen = cls.get_mario_location_on_screen(ram)

        #下面for循环的迭代坐标
        x_start = mario_level.x - mario_screen.x
        y_start = 0

        enemies = cls.get_enemy_locations(ram)

        row,col = 0,0
        # 遍历内存，并读取釉面信息
        for y_pos in range(y_start, 240, 16):
            for x_pos in range(x_start, x_start + 256, 16):
                loc = (row, col)
                # PPU is there, so no tile is there
                if row < 2:
                    # 马里奥里地面高度是1，地面下面是空的
                    tiles[loc] =  StaticTileType.Empty
                else:            
                    try:
                        tile = cls.get_tile(x_pos, y_pos, ram)
                        tiles[loc] = StaticTileType(tile)
                    except:
                        tiles[loc] = StaticTileType.Fake
                    # 将设定范围内的敌人存入数组中
                    for enemy in enemies:
                        ex = enemy.location.x + 8
                        ey = enemy.location.y + 8
                        # Since we can only discriminate within 8 pixels, if it falls within this bound, count it as there
                        if abs(x_pos - ex) <=8 and abs(y_pos - ey) <=8:
                            tiles[loc] = EnemyType.Generic_Enemy
                # Next col
                col += 1
            # Move to next row
            col = 0
            row += 1

        # 将马里奥放入釉面数组中对应位置
        tiles[cls.get_mario_row_col(ram)] = DynamicTileType.Mario

        return tiles

    @classmethod
    def get_mario_row_col(cls, ram):
        x, y = cls.get_mario_location_on_screen(ram)
        # Adjust 16 for PPU
        y = ram[cls.RAMLocations.Player_Y_Position_Screen_Offset.value] + 16
        x += 12
        col = x // 16
        row = (y - 0) // 16
        return (row, col)

    # 根据x,y计算其在内存中的位置，并返回
    @classmethod
    def get_tile(cls, x, y, ram, group_non_zero_tiles=True):
        page = (x // 256) % 2
        sub_x = (x % 256) // 16
        sub_y = (y - 32) // 16

        if sub_y not in range(13):
            return StaticTileType.Empty.value

        # 0x500=1280 说明前1280字节都不属于地图信息，page双缓冲：一页游玩一页生成
        addr = 0x500 + page*208 + sub_y*16 + sub_x
        if group_non_zero_tiles and ram[addr]:
            return StaticTileType.Fake.value
        return ram[addr]
