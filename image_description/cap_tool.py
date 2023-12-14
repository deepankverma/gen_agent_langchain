##caption tool not providing good resultsfrom blip-image-captioning model hence can be removed, instead caption can be generated with the help of placesCNN

# from image_description.img_desc import caption
from image_description.places365.run_placesCNN_unified import places365
from langchain.tools import BaseTool  ## for custom transformer tool

desc = (
    "use this tool when given the URL of the image and asked about the image caption only."
    "It will return a list of scene categories which will help the agent decide which direction is more prominent in acheiving the goal."
)

class outputs_cap_tools(BaseTool):
    name = "Cap"
    description = desc

    def _run(self, url):

        # cap = caption(url)
        scene_cat, scene_attr = places365(url)

        # comb = (scene_cat)

        output_string = f"Scene categories ==  {scene_cat}"


        return output_string
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    


    