from PIL import Image
import pydotplus as pdp
from GBDT.decision_tree import Node
import os
import matplotlib.pyplot as plt
def print_tree(tree,screen,max_depth,iter):
    root = tree.root_node
    return solve(root,screen,max_depth,iter)

def plot_all_trees(numberOfTrees):
    if numberOfTrees/ 3 -int(numberOfTrees/ 3) > 0.000001:
        rows = int(numberOfTrees/ 3)+1
    else:
        rows = int(numberOfTrees/ 3)
    plt.figure(1,figsize=(30,20))
    plt.axis('off')
    try:
        for index in range(1, numberOfTrees + 1):
            path = os.path.join('results','NO.{}_tree.png'.format(index))
            plt.subplot(rows, 3, index)

            img = Image.open(path)
            img = img.resize((1000,800),Image.ANTIALIAS)
            plt.axis('off')
            plt.title('NO.{} tree'.format(index))
            plt.imshow(img)
        plt.savefig('results/all_trees.png',dpi=300)
        plt.show()
        image_compose(numberOfTrees)

    except Exception as e:
        raise e

def image_compose(numberOfTrees):
    png_to_compose = []
    for index in range(1,numberOfTrees+1):
        png_to_compose.append('NO.{}_tree.png'.format(index))
    try:
        path = os.path.join('results', png_to_compose[0])
        shape = Image.open(path).size
    except Exception as e:
        raise e
    IMAGE_WIDTH = shape[0]
    IMAGE_HEIGET = shape[1]
    IMAGE_COLUMN = 3

    if len(png_to_compose)/IMAGE_COLUMN - int(len(png_to_compose)/IMAGE_COLUMN) >0.0000001:
        IMAGE_ROW = int(len(png_to_compose)/IMAGE_COLUMN)+1
    else:
        IMAGE_ROW = int(len(png_to_compose) / IMAGE_COLUMN)
    to_image = Image.new('RGB',(IMAGE_COLUMN*IMAGE_WIDTH,IMAGE_ROW*IMAGE_HEIGET),'#FFFFFF')
    for y in  range(IMAGE_ROW):
        for x in range(IMAGE_COLUMN):
            if y*IMAGE_COLUMN+x+1>len(png_to_compose):
                break
            path = os.path.join('results','NO.'+str(y*IMAGE_COLUMN+x+1)+'_tree.png')
            from_image = Image.open(path)
            to_image.paste(from_image,(x*IMAGE_WIDTH,y*IMAGE_HEIGET))

    to_image.save('results/all_trees_high_quality.png')








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
    for depth in range(max_depth):
        for nodepair in res:
            if nodepair[0].deep == depth:
                p, c = nodepair[0], nodepair[1]
                pname = str(list(nodes.keys())[list(nodes.values()).index(p)])
                cname = str(list(nodes.keys())[list(nodes.values()).index(c)])

                edges = edges + pname + '->' + cname + '[label=\"' + str(p.split_feature) + (
                    '<' if p.left_child == c else '>=') + str(p.split_value) + '\"]' + ';\n'
                node = node + pname + '[width=1,height=0.5,color=lemonchiffon,style=filled,shape=ellipse,label=\"id:' + str(
                    [i for i in range(len(p.data_index)) if p.data_index[i] is True]) + '\"];\n' + \
                       cname + '[width=1,height=0.5,color=lemonchiffon,style=filled,shape=ellipse,label=\"id:' + str(
                    [i for i in range(len(c.data_index)) if c.data_index[i] is True]) + '\"];\n'
                       # ('\npredict_value:' + str("{:.4f}".format(c.predict_value)) if c.is_leaf else '') + '\"];\n'
                if c.is_leaf:
                    edges = edges+cname+'->'+cname+'p[style=dotted];\n'
                    node = node+ cname+'p[width=1,height=0.5,color=lightskyblue,style=filled,shape=box,label=\"'+str("{:.4f}".format(c.predict_value))+'\"];\n'


            else:
                continue
        dot = '''digraph g {\n''' + edges + node + '''}'''
        graph = pdp.graph_from_dot_data(dot)
        graph.write_png('results/NO.{}_tree.png'.format(iter))
        img = Image.open('results/NO.{}_tree.png'.format(iter))
        img = img.resize((1024, 700), Image.ANTIALIAS)

        plt.ion()
        plt.figure(1,figsize=(30,20))
        plt.axis('off')
        plt.title('NO.{} tree'.format(iter))
        plt.rcParams['figure.figsize'] = (30.0, 20.0)
        plt.imshow(img)
        plt.pause(0.02)
        plt.close()
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

if __name__ =="__main__":
    # plot_all_trees(10)
    image_compose(10)

