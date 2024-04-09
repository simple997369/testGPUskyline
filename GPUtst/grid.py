import os, sys
sys.path.append(os.path.abspath(os.pardir))

from data.dataClass import batchImport

class Point(object):
    def __init__(self, *args):
        """
        Constructs a `Point` object.

        """
        if len(args[0]) < 2 or len(args[0]) > 10:
            raise ValueError("Point must have between 2 and 10 dimensions")
        self.point = args
        print(self.point)
    
    def __repr__(self):
        return 'Point'f"{' Point'.join(str(p) for p in self.point)}"

class GridIndex:
    """
    Constructs a `GridIndex` object.

    """
    def __init__(self, grid_size, dim, count, ps):
        self.grid_size = grid_size
        self.dim = dim
        self.count = count
        self.ps = ps
        self.grid = {}
        self.sorted_axis = []
        self.all = []

    def calculate_grid_index(self, *point):
        self.point = point
        for i in range(len(self.point)):
            _axis = self._calculate_grid_index(*self.point[i])
            # print(_axis)
            if _axis not in self.grid:
                self.grid[_axis] = []
            self.grid[_axis].append(int(self.point[i][1]))

    def _calculate_grid_index(self, *args):
        all_axis = args 
        temp = list(all_axis)[3:]
        temp = [int(x * (2**self.grid_size)) for x in temp]
        # print(temp)
        self.all.append(temp)
        # print(self.all)
        all_axis = tuple(temp)
        return all_axis
    
    def print_grid(self):
        # for grid_index, points in self.grid.items():
        #     print("Grid Index:", grid_index)
        #     print("Points:", points)
        print(self.sorted_axis)
        # print(self.grid)

    def sort_grid(self):
        # print(sorted(self.grid.keys()))
        for i in range(self.dim):
            self.sorted_axis.append(sorted(self.grid.keys(), key=lambda x: x[i]))
        

    # def insert(self, point):
    #     x, y = point
    #     grid_x, grid_y = self._calculate_grid_index(x, y)
    #     if (grid_x, grid_y) not in self.grid:
    #         self.grid[(grid_x, grid_y)] = []
    #     self.grid[(grid_x, grid_y)].append(point)

    # def query(self, query_point):
    #     x, y = query_point
    #     grid_x, grid_y = self._calculate_grid_index(x, y)
    #     if (grid_x, grid_y) in self.grid:
    #         return self.grid[(grid_x, grid_y)]
    #     else:
    #         return []

    


if __name__ == '__main__':

    dim = 2
    count = 5
    ps = 2
    threshold = 0.6
    host_data, Max = batchImport('anticor_'+str(dim)+'d_'+str(count)+'_'+str(ps)+'.txt', count, dim, ps)

    # print(*host_data)

    # Example usage:
    # point = Point(*host_data[:,3:])
    # print(point)

    grid_size = 2
    grid_index = GridIndex(grid_size, dim, count, ps)
    grid_index.calculate_grid_index(*host_data)
    grid_index.print_grid()
    grid_index.sort_grid()






    # 插入數據點
    # grid_index.insert(point)
    # grid_index.insert((15, 15))
    # grid_index.insert((25, 25))
    # grid_index.insert((25, 26))

    # # 查詢特定區域
    # result = grid_index.query((12, 12))
    # print("Query Result:", result)  # [(5, 5), (15, 15)]

    # # 打印grid
    # grid_index.print_grid()



