# --------------------------------------------------------------------------------
# Copyright (c) [2024] by Charactr Company
#
# This script is proprietary software owned by Charactr Company. It is distributed
# under exclusive license and may only be used, copied, modified, or distributed
# with express written permission from Charactr Company. Unauthorized use, copying,
# modification, or distribution is strictly prohibited.
#
# For license inquiries, contact Charactr Company directly.
#
# DISCLAIMER: This software is provided by Charactr Company "as is" and any
# expressed or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are
# disclaimed. In no event shall Charactr Company or its contributors be liable
# for any direct, indirect, incidental, special, exemplary, or consequential
# damages (including, but not limited to, procurement of substitute goods or
# services; loss of use, data, or profits; or business interruption) however
# caused and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of the use
# of this software, even if advised of the possibility of such damage.
#
# Created by: [Arslan Ali]
# Created on: [Apr 2024]
# --------------


from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a highly detailed portrait of a indian man, 8k"
image = pipe(prompt).images[0]  
    
image.save("indian.png")