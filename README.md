# NeRF custom implementation

NeRF (Neural Radiance Fields) is a method of generating different views of a 3D object, from very sparse input views. In this repo, I'll be implementing it myself. 

### The major things I'll be making are:

The MLP network that takes in any point in 3D space, along with the direction the camera is facing, to give the volume density at that point, and the RGB color at that point

The ray generator and ray marcher, which will create a ray from the camera for every single pixel of the output render, and break the ray into small segments respectively

The volume renderer which will use these rays, along with some simple equations to render the output equation. The volume renderer will be completely
differentiable, meaning I can run a loss function on the generated image, and the actual image for this view.

I will train all these on the Lego Bulldozer Tiny dataset, which is what the original NeRF paper trained on.

After this, I will script my own blender add-on. The user will set a camera position and angle, and hit a button. The NeRF will then render the scene for them


# Progress Update 1

the basic implementation, trained on lego bulldozer tiny dataset is done. next i'll be making a blender addon to generate custom datasets and train on them. here are the results so far:
<img width="1666" height="787" alt="image" src="https://github.com/user-attachments/assets/4e9b1dc5-b027-45cb-9e0d-bf7ba65f88fc" />
<img width="1663" height="832" alt="image" src="https://github.com/user-attachments/assets/8c96ddf9-0880-493d-81b9-a948ac11caf3" />
<img width="1693" height="837" alt="image" src="https://github.com/user-attachments/assets/80f13d11-86c4-4535-bd29-e373b26de2ae" />


here are some of the final renders:
<img width="369" height="369" alt="rendered_epoch_1_img_100" src="https://github.com/user-attachments/assets/e2db0832-f525-4975-937b-58850585ad38" />
<img width="369" height="369" alt="rendered_epoch_1_img_94" src="https://github.com/user-attachments/assets/f3883b2f-cbf7-45fe-b8a7-23ff9c23784c" />
