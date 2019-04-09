import pydotplus as pdp
from PIL import Image
import pydotplus as pdp
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from GBDT.decision_tree import Node
def printtree(tree):
    root = tree.root_node
    solve(root)
    a=1

def bianli(root:Node, res: list):
    if (root == None):
        return
    if (root.left_child != None):
        res.append([root, root.left_child])
        bianli(root.left_child, res)
    if (root.right_child != None):
        res.append([root, root.right_child])
        bianli(root.right_child, res)


    # if root.left_child == None and root.right_child == None:
    #     return str(root.data_index)
    # nownode = str(root.data_index)
    # data = ''
    # if root.left_child != None:
    #     data = data+'['+nownode+'l->'+bianli(root.left_child)+']'
    # if root.right_child != None:
    #     data = data+'['+nownode+'l->'+bianli(root.right_child)+']'
    # return data

def solve(root):
    res = []
    bianli(root, res)

    print(res)
    nodes = {}
    index = 0
    for i in res:
        p,c =i[0],i[1]
        if p not  in  nodes.values():
            nodes[index] = p
            index = index+1
        if c not  in  nodes.values():
            nodes[index] = c
            index = index+1
    edges = ''
    ss = ''
    for i in res:
        p,c = i[0],i[1]
        pname = str(list(nodes.keys())[list(nodes.values()).index(p)])
        cname = str(list(nodes.keys())[list(nodes.values()).index(c)])
        print(pname,cname)
        edges = edges+pname+'->'+cname+';\n'
        ss = ss + pname+'[shape=box,label=\"'+str(p.data_index)+'\"];\n'+cname + '[shape=box,label=\"' + str(c.data_index) + '\"];\n'
    dot = '''digraph g {\n'''+edges+ss+'''}'''
    print(dot)
    graph = pdp.graph_from_dot_data(dot)
    graph.write_png('1.png')
    img = Image.open('1.png')
    img.show()
    # lena = mpimg.imread('1.png')
    # plt.imshow(lena)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()


    # print(nodes)






