class ClassMapping():
    def __init__(self):
        # self.class_mapping
        # 0: 마리오
        # 1: 적
        # 2: 아이템, 코인블록, 토관, 깃발, 깰 수 있는 블록
        # 3: 밟을 수 있는 블록

        # Mario_small = 0
        # Mario_big = 1
        # Mario_fire = 2
        # Enemy = 3


        # Mushroom = 4
        # Flower = 5
        # Star = 6
        # LifeUp = 7


        # # Empty = 0x00
        # Ground = 8
        # Top_Pipe1 = 9
        # Top_Pipe2 = 10
        # Bottom_Pipe1 = 11
        # Bottom_Pipe2 = 12
        # Pipe_Horizontal = 13


        # Flagpole_Top =  14
        # Flagpole = 15
        # Coin_Block = 16
        # Coin_Block_End = 17
        # Coin = 18

        # Breakable_Block = 19

        self.class_mapping = {
            0 : [0,1,2],
            1 : [3],
            2 : [4,5,6,7,13,14,15,16,18,19],
            3 : [8,9,10,11,12,13,16,17,19]
        }

        # self.class_mapping = {
        #     0 : [0,1,2],
        #     1 : [3],
        #     2 : [4,5,6,7,9,10,13,14,15,16,18,19],
        #     3 : [8,9,10,11,12,13,16,17,19]
        # }

        self.rgb_mapping = {
            0: (100,100,100),
            1: (255,0,0),
            2: (0,255,0),
            3: (0,0,255),
        }


    def get_group_id(self, class_id):
        group_id = None
        for key, class_list in self.class_mapping.items():
            if class_id in class_list:
                group_id = key
                break
        
        return group_id
    
    def get_color_by_group_id(self, group_id):
        return self.rgb_mapping[group_id]

    # def get_color_by_class_id(self, class_id):
    #     group_id = self.get_group_id(class_id)
    #     return self.rgb_mapping[group_id]
    
        