from PIL import Image
import pydotplus as pdp
from GBDT.decision_tree import Node
import pygame
import time
import os
import matplotlib.pyplot as plt
def print_tree(tree,screen,max_depth,iter):
    root = tree.root_node
    return solve(root,screen,max_depth,iter)

def plot_all_trees():
    png_list = os.listdir('trees_png')
    rows = int(len(png_list) / 3)
    plt.figure(1)
    plt.axis('off')
    plt.rcParams['figure.figsize'] = (30.0, 20.0)
    for index in range(1, len(png_list) + 1):
        path = os.path.join('trees_png', png_list[index - 1])
        if os.path.isfile(path) and not path.find('trees'):
            plt.subplot(rows + 1, 3, index)
            img = Image.open(path)
            plt.axis('off')
            plt.title('NO.{} tree'.format(index))
            plt.imshow(img)
    plt.savefig('trees_png/trees.png',dpi=300)
    plt.show()
def traversal(root: Node, res: list):
    if root is None:
        return
    if root.left_child is not None:
        res.append([root, root.left_child])
        traversal(root.left_child, res)
    if root.right_child is not None:
        res.append([root, root.right_child])
        traversal(root.right_child, res)
def traversal2(root:Node,res:list):
    outList = []
    queue = [root]
    while queue != [] and root:
        outList.append(queue[0].data_index)
        if queue[0].left_child != None:
            queue.append(queue[0].left_child)
            res.append([queue[0],queue[0].left_child])
        if queue[0].right_child != None:
            queue.append(queue[0].right_child)
            res.append([queue[0], queue[0].right_child])
        queue.pop(0)
    print(outList)
    print(res)



def solve(root,screen,max_depth,iter):
    res = []
    traversal2(root, res)
    nodes = {}
    index = 0
    for i in res:
        p, c = i[0], i[1]
        if p not in nodes.values():
            nodes[index] = p
            index = index+1
        if c not in nodes.values():
            nodes[index] = c
            index = index+1
    edges = ''
    node = ''
    nodes_in_depth = []
    for depth in range(max_depth):
        index = 0
        for nodepair in res:
            if nodepair[0].deep == depth:
                p, c = nodepair[0], nodepair[1]
                pname = str(list(nodes.keys())[list(nodes.values()).index(p)])
                cname = str(list(nodes.keys())[list(nodes.values()).index(c)])

                edges = edges + pname + '->' + cname + '[label=\"' + str(p.split_feature) + (
                    '<' if p.left_child == c else '>=') + str(p.split_value) + '\"]' + ';\n'
                node = node + pname + '[shape=ellipse,label=\"data_index:' + str(
                    [i for i in range(len(p.data_index)) if p.data_index[i] is True]) \
                       + '\nsplit_feature:' + str(p.split_feature) + '\nsplit_value:' + str(p.split_value) + '\"];\n' + \
                       cname + '[shape=ellipse,label=\"data_index:' + str(
                    [i for i in range(len(c.data_index)) if c.data_index[i] is True]) + \
                       ('\npredict_value:' + str("{:.4f}".format(c.predict_value)) if c.is_leaf else '') + '\"];\n'

                index = res.index(nodepair)
            else:
                continue
        dot = '''digraph g {\n''' + edges + node + '''}'''
        graph = pdp.graph_from_dot_data(dot)
        graph.write_png('trees_png/NO.{}_tree.png'.format(iter))
        img = Image.open('trees_png/NO.{}_tree.png'.format(iter))
        img = img.resize((1024, 700), Image.ANTIALIAS)

        plt.ion()
        plt.figure(1)
        plt.axis('off')
        plt.title('NO.{} tree'.format(iter))
        plt.rcParams['figure.figsize'] = (30.0, 20.0)
        # plt.rcParams['savefig.dpi'] = 300  # 图片像素
        # plt.rcParams['figure.dpi'] = 300  # 分辨率
        plt.imshow(img)
        plt.pause(0.1)
        plt.close()
        # img.save('1.png')

        # img = pygame.image.load('1.png')
        # # img = pygame.transform.scale(img,(1024,700))
        # screen.blit(img, (0, 0))
        # pygame.display.update()
        # time.sleep(2)
    return screen



    # for i in res:
    #     p, c = i[0], i[1]
    #     pname = str(list(nodes.keys())[list(nodes.values()).index(p)])
    #     cname = str(list(nodes.keys())[list(nodes.values()).index(c)])
    #
    #     edges = edges+pname+'->'+cname+'[label=\"'+str(p.split_feature) + ('<' if p.left_child ==c else '>=') + str(p.split_value)+'\"]'+';\n'
    #     node = node + pname+'[shape=ellipse,label=\"data_index:'+str([i for i in range(len(p.data_index)) if p.data_index[i] is True])\
    #          +'\nsplit_feature:'+str(p.split_feature)+'\nsplit_value:'+str(p.split_value)+'\"];\n'+\
    #          cname + '[shape=ellipse,label=\"data_index:' + str([i for i in range(len(c.data_index)) if c.data_index[i] is True]) + \
    #            ('\npredict_value:'+str("{:.4f}".format(c.predict_value)) if c.is_leaf else '')+'\"];\n'
    #     dot = '''digraph g {\n''' + edges + node + '''}'''
    #     graph = pdp.graph_from_dot_data(dot)
    #     graph.write_png('1.png')
    #     img = Image.open('1.png')
    #     img = img.resize((1024,700),Image.ANTIALIAS)
    #     img.save('1.png')
        # img = pygame.image.load('1.png')
        # screen.blit(img, (0, 0))
        # pygame.display.flip()
        # time.sleep(2)

    # lena = mpimg.imread('1.png')
    # plt.imshow(lena)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
    # print(nodes)


