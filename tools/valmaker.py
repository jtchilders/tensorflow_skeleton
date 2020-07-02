import glob,os,shutil
import xml.etree.ElementTree as ET


# make Data/CLS-LOC/val/* images organized in label folders
def mv_val_images():
   xmlpath = '/gpfs/jlse-fs0/projects/datascience/parton/data/imagenet/ILSVRC/Annotations/CLS-LOC/val'
   imgpath = '/gpfs/jlse-fs0/projects/datascience/parton/data/imagenet/ILSVRC/Data/CLS-LOC/val'
   infiles = glob.glob(imgpath + '/*.JPEG')

   print(len(infiles),'files to process')
   for filename in infiles:
      xmlfilename = os.path.join(xmlpath,os.path.basename(filename)).replace('.JPEG','.xml')
      # print(xmlfilename)
      tree = ET.parse(xmlfilename)
      root = tree.getroot()

      label = root.find('object').find('name').text
      # print(xmlfilename,label)

      new_imgpath = imgpath + '/'  + label
      # print(new_imgpath)
      if not os.path.exists(new_imgpath):
         os.mkdir(new_imgpath)
      shutil.move(filename,new_imgpath  + '/' + filename.split('/')[-1])


def mv_val_annots():
   xmlpath = '/gpfs/jlse-fs0/projects/datascience/parton/data/imagenet/ILSVRC/Annotations/CLS-LOC/val'
   infiles = glob.glob(xmlpath + '/*.xml')

   print(len(infiles), 'files to process')
   for filename in infiles:
      # print(filename)
      tree = ET.parse(filename)
      root = tree.getroot()

      label = root.find('object').find('name').text
      # print(filename,label)

      new_imgpath = xmlpath + '/' + label
      # print(new_imgpath + '/' + filename.split('/')[-1])
      if not os.path.exists(new_imgpath):
         os.mkdir(new_imgpath)
      shutil.move(filename, new_imgpath + '/' + filename.split('/')[-1])

mv_val_annots()