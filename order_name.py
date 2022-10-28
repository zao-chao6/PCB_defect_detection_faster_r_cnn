#...........................#
#对文件夹中的文件进行重命名
#...........................#
import os
import xml
from xml.dom import minidom
import xml.etree.cElementTree as ET

def myrename(file_path):
    file_list=os.listdir(file_path)
    for i,fi in enumerate(file_list):
        old_dir=os.path.join(file_path,fi)
        print('wenjianmingzi :',old_dir)
        # 删除名字中的空格
        new_name = fi.replace(" ", "_")
        print("新名字为：",new_name)

        # # 顺序命名
        # # new_name=str(i+1)+"."+str(fi.split(".")[-1])
        new_dir=os.path.join(file_path,new_name)
        try:
            os.rename(old_dir,new_dir)
        except Exception as e:
            print(e)
            print("Failed!")
        else:
            print("SUcess!")


#...........................#
#对xml文件内的filename和path名进行重命名
#...........................#

def xml_name(xmlpath):
    files = os.listdir(xmlpath)  # 得到文件夹下所有文件名称
    count = 0
    for xmlFile in files:  # 遍历文件夹
        if not os.path.isdir(xmlFile):  # 判断是否是文件夹,不是文件夹才打开
            name1 = xmlFile.split('.')[0]
            dom = xml.dom.minidom.parse(xmlpath + '/' + xmlFile)
            root = dom.documentElement
            #filename重命名
            newfilename = root.getElementsByTagName('filename')
            t=newfilename[0].firstChild.data = name1 + '.jpg'
            print('t:',t )
            #path重命名
            newpath = root.getElementsByTagName('path')
            t1=newpath[0].firstChild.data =xmlpath +'\\'+ name1 +'.jpg'
            print('t1:',t1 )

            with open(os.path.join(xmlpath, xmlFile), 'w',) as fh:
                print('fh:',fh )
                dom.writexml(fh)
                print('写入name/pose OK!')
            count = count + 1


# 删除xml文件中显示的版本号
def delete_xmlversion(xmlpath,savedir):
    
    files = os.listdir(xmlpath)
    for ml in files:
        if '.xml' in ml:
            fo = open(savedir + '/' + '{}'.format(ml), 'w', encoding='utf-8')
            print('{}'.format(ml))
            fi = open(xmlpath + '/' + '{}'.format(ml), 'r')
            content = fi.readlines()
            for line in content:
                # line = line.replace('a', 'b')        # 例：将a替换为b
                line = line.replace('<?xml version="1.0" ?>', '')
                # line = line.replace('<folder>测试图片</folder>', '<folder>车辆图片</folder>')
                # line = line.replace('<name>class1</name>', '<name>class2</name>')
                fo.write(line)
            fo.close()
            print('替换成功')


#删除xml文件中部分不要的标签信息
def Delete_part_information_xml(path_root,xy_classes):
    for anno_path in path_root:
        xml_list=os.listdir(anno_path)
        print("打开{}文件".format(xml_list))
        for annoxml in xml_list:
            path_xml=os.path.join(anno_path,annoxml)
            print('保存文件路径为{}'.format(path_xml))
            tree =ET.parse(path_xml)
            root=tree.getroot()

            for child in root.findall('object'):
                name = child.find('name').text
                if not name in xy_classes:
                    root.remove(child)
            print(annoxml)
            tree.write(os.path.join(r'F:\Desktop\PCB_code\PCB_DataSet\Annotations—new', annoxml))  #处理结束后保存的路径




if __name__=="__main__":
    file_path=r"F:\Desktop\PCB_code\date_set\new_data"   #完整路径+文件名
    # xmlpath="F:\\桌面\\PCB_code\\date_set\\Image_label_source"
    # savedir = r'F:\桌面\PCB_code\date_set\3' #删除xml文件中显示的版本号后存放文件位置
    # xmlpath=r'F:\桌面\PCB_code\date_set\label'
    myrename(file_path)        #图片重命名文件

    #对xml文件中的名字进行修改
    # myrename(xmlpath)          #1、xml文件名重命名
    # xml_name(xmlpath)          #2、xml文件内的filename和path重命名
    # delete_xmlversion(xmlpath,savedir)  #删除经过xml重命名后文件内的版本号

    #删除xml文件中部分不要的标签信息
    path_root=r'F:\Desktop\PCB_code\PCB_DataSet\Annotations'
    xy_classes=['Speaker',"Bat","2USB","Rj45+2USB","Cap_cross","Cap_blue_black","Jumper04p",
                "Jumper10p", "HDD","Power08p","Power04p","Power24p"]  
    Delete_part_information_xml(path_root,xy_classes)

