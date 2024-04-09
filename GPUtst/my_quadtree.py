class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"
    
class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def contains(self, point):
        return(self.x <= point.x <= self.w and
               self.y <= point.y <= self.h)

        # return (point.x >= self.x - self.w / 2 and
        #         point.x <= self.x + self.w / 2 and
        #         point.y >= self.y - self.h / 2 and
        #         point.y <= self.y + self.h / 2)


class QuadTreeNode:
    def __init__(self, boundary, capacity):
        self.boundary = boundary # 边界框
        self.capacity = capacity # 节点容量
        self.points = [] # 存储的点
        self.subdivided = False # 是否已经细分成四个子节点
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None

    def insert(self, point):
        if not self.boundary.contains(point):
            return False

        if len(self.points) < self.capacity:
            self.points.append(point)
            return True
        else:
            if not self.subdivided:
                self.subdivide()

            if self.northwest.insert(point):
                return True
            elif self.northeast.insert(point):
                return True
            elif self.southwest.insert(point):
                return True
            elif self.southeast.insert(point):
                return True

    def subdivide(self):
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.w
        h = self.boundary.h

        ne_boundary = Rectangle((x + w) / 2, (y + h) / 2, w , h )
        self.northeast = QuadTreeNode(ne_boundary, self.capacity)

        nw_boundary = Rectangle(x , (y + h) / 2, (x + w) / 2, h )
        self.northwest = QuadTreeNode(nw_boundary, self.capacity)

        se_boundary = Rectangle((x + w) / 2, y , w , (y + h) / 2)
        self.southeast = QuadTreeNode(se_boundary, self.capacity)

        sw_boundary = Rectangle(x , y , (x + w) / 2, (y + h) / 2)
        self.southwest = QuadTreeNode(sw_boundary, self.capacity)

        self.subdivided = True

    def print_tree(self, depth=0):
        print('  ' * depth + f'Node at ({self.boundary.x}, {self.boundary.y}), Points: {len(self.points)}')
        if self.subdivided:
            self.northwest.print_tree(depth + 1)
            self.northeast.print_tree(depth + 1)
            self.southwest.print_tree(depth + 1)
            self.southeast.print_tree(depth + 1)

    def print_points(self):
        # print(f"Points in Node at ({self.boundary.x}, {self.boundary.y}):")
        for point in self.points:
            print(f"({point.x}, {point.y})")
        
        if self.subdivided:
            self.northwest.print_points()
            self.northeast.print_points()
            self.southwest.print_points()
            self.southeast.print_points()

    def find_node(self, x, y):
        if not self.boundary.contains(Point(x, y)):
            return None
        
        if not self.subdivided:
            return self

        if self.northwest.boundary.contains(Point(x, y)):
            return self.northwest.find_node(x, y)
        elif self.northeast.boundary.contains(Point(x, y)):
            return self.northeast.find_node(x, y)
        elif self.southwest.boundary.contains(Point(x, y)):
            return self.southwest.find_node(x, y)
        elif self.southeast.boundary.contains(Point(x, y)):
            return self.southeast.find_node(x, y)

        return None


class QuadTree:

    node_class = QuadTreeNode
    point_class = Point

    def __init__(self, boundary, capacity):
        self.root = QuadTreeNode(boundary, capacity)

    def insert(self, point):
        return self.root.insert(point)
    
    def find_node(self, x, y):
        return self.root.find_node(x, y)

if __name__ == "__main__":
    boundary = Rectangle(0, 0, 400, 400)
    print(boundary)
    quadtree = QuadTree(boundary, 4)
    points = [Point(100, 100), Point(200, 200), Point(300, 300), Point(150, 150)]

    for point in points:
        quadtree.insert(point)

    # 打印 Quadtree 结构
    # quadtree.root.print_tree()

    # quadtree.root.print_points()

    x = 1
    y = 1
    node = quadtree.find_node(x, y)
    if node:
        print(f"Points in Node at ({x}, {y}):")
        node.print_points()
    else:
        print(f"No node found at ({x}, {y})")


