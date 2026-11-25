from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageOps

def prepare(source, target, key, box, label, source_id):
    with Image.open(source) as photo:
        image = ImageOps.exif_transpose(photo).convert("L")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
        raise ValueError(f"Invalid box for {source}: {box}")
    span = max(x2-x1, y2-y1)
    side = math.ceil(span * 1.5)
    left = math.floor((x1+x2-side)/2)
    top = math.floor((y1+y2-side)/2)
    background = Image.new("L", (side, side), 128)
    visible = (max(left,0),max(top,0),min(left+side,image.width),min(top+side,image.height))
    background.paste(image.crop(visible), (visible[0]-left,visible[1]-top))
    clean = background.resize((200,200),Image.Resampling.LANCZOS)
    coords = tuple((value-origin)*200/side for value, origin in zip(box,(left,top,left,top)))
    overlay = clean.convert("RGB")
    ImageDraw.Draw(overlay).rectangle(
        (math.floor(coords[0]),math.floor(coords[1]),math.ceil(coords[2])-1,math.ceil(coords[3])-1),
        outline="lime",width=1)
    for name in ("raw","processed","annotated"):
        (target/name).mkdir(parents=True,exist_ok=True)
    raw = target/"raw"/source.name
    if not raw.exists():
        shutil.copyfile(source,raw)
    clean.save(target/"processed"/f"{key}.png")
    overlay.save(target/"annotated"/f"{key}.png")
    return [key,source_id,source.name,label,*box,*coords,left,top,side]

def save_manifest(target, rows):
    target.mkdir(parents=True,exist_ok=True)
    with (target/"boxes.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id","source_id","raw_file","label","raw_x1","raw_y1","raw_x2","raw_y2",
                         "x1","y1","x2","y2","crop_left","crop_top","crop_side"])
        writer.writerows(rows)

def gc10(annotations, image_dir, target, limit=20):
    data = json.loads(annotations.read_text())
    categories = {c["id"]:c["name"].strip().casefold() for c in data["categories"]}
    candidates = defaultdict(list)
    for item in data["annotations"]:
        candidates[item["image_id"]].append(item)
    rows = []
    for image in sorted(data["images"],key=lambda item:str(item["id"])):
        items = candidates[image["id"]]
        if len(items)!=1 or items[0].get("iscrowd",0):
            continue
        item = items[0]
        if categories[item["category_id"]]!="inclusion":
            continue
        x,y,w,h = item["bbox"]
        source = image_dir/image["file_name"]
        if source.is_file() and w>=8 and h>=8:
            rows.append(prepare(source,target,f"gc10_{len(rows)+1:03}",(x,y,x+w,y+h),"inclusion",image["id"]))
        if len(rows)>=limit:
            break
    save_manifest(target,rows)
    return rows

def severstal(annotations,image_dir,review,target):
    reviewed = set()
    with review.open(newline="",encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            reviewed.add((item["ImageId"],str(item["ClassId"])))
    rows = []
    with annotations.open(newline="",encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            if "ImageId_ClassId" in item:
                filename,class_id = item["ImageId_ClassId"].rsplit("_",1)
            else:
                filename,class_id = item["ImageId"],item["ClassId"]
            if (filename,class_id) not in reviewed or not item["EncodedPixels"].strip():
                continue
            source = image_dir/filename
            with Image.open(source) as image:
                width,height = image.size
            runs = np.fromstring(item["EncodedPixels"],sep=" ",dtype=np.int64).reshape(-1,2)
            flat = np.zeros(width*height,dtype=bool)
            for start,length in runs:
                if not (1<=start and length>0 and start-1+length<=flat.size):
                    raise ValueError(f"Invalid RLE for {filename}")
                flat[start-1:start-1+length]=True
            yy,xx = np.where(flat.reshape((height,width),order="F"))
            box = (int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
            rows.append(prepare(source,target,f"severstal_{len(rows)+1:03}",box,
                                f"severstal_class_{class_id}",filename))
    save_manifest(target,rows)
    return rows
