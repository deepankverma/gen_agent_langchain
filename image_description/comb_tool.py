from image_description.segment import segmentation
from image_description.img_desc import caption
from image_description.places365.run_placesCNN_unified import places365
from image_description.obj_det import det
from langchain.tools import BaseTool  ## for custom transformer tool

desc = (
    "use this tool when given the URL of the image and asked about the summary regarding the content of the image."
    "It will return a tuple of string and dictionary of overall caption and detected elements, paired with the corresponding percentage, respectively."

)

class outputs_all_tools(BaseTool):
    name = "Seg+Cap+Class"
    description = desc

    def _run(self, img):

        # cap = caption(url)
        seg = segmentation(img)
        scene_cat, scene_attr = places365(img)
        dets = det(img)

        # comb = (seg, scene_cat, scene_attr, dets)

        output_string = f"Segmentation results ==  {seg}, Scene categories ==  {scene_cat}, Scene attributes == {scene_attr}, Object detections == {dets}."

        return output_string
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    


    