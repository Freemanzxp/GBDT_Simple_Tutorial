from PIL import Image
import pydotplus as pdp
from GBDT.decision_tree import Node


def printtree(tree):
    root = tree.root_node
    solve(root)


def traversal(root: Node, res: list):
    if root is None:
        return
    if root.left_child is not None:
        res.append([root, root.left_child])
        traversal(root.left_child, res)
    if root.right_child is not None:
        res.append([root, root.right_child])
        traversal(root.right_child, res)


def solve(root):
    res = []
    traversal(root, res)

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
    for i in res:
        p, c = i[0], i[1]
        pname = str(list(nodes.keys())[list(nodes.values()).index(p)])
        cname = str(list(nodes.keys())[list(nodes.values()).index(c)])

        edges = edges+pname+'->'+cname+'[label=\"'+str(p.split_feature) + ('<' if p.left_child ==c else '>=') + str(p.split_value)+'\"]'+';\n'
        node = node + pname+'[shape=ellipse,label=\"data_index:'+str([i for i in range(len(p.data_index)) if p.data_index[i] is True])\
             +'\nsplit_feature:'+str(p.split_feature)+'\nsplit_value:'+str(p.split_value)+'\"];\n'+\
             cname + '[shape=ellipse,label=\"data_index:' + str([i for i in range(len(c.data_index)) if c.data_index[i] is True]) + \
               ('\npredict_value:'+str("{:.4f}".format(c.predict_value)) if c.is_leaf else '')+'\"];\n'
    dot = '''digraph g {\n'''+edges+node+'''}'''
    graph = pdp.graph_from_dot_data(dot)
    graph.write_png('1.png')
    img = Image.open('1.png')
    img.show()
    # lena = mpimg.imread('1.png')
    # plt.imshow(lena)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
    # print(nodes)


