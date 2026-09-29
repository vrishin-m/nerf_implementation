# NeRF custom implementation

NeRF (Neural Radiance Fields) is a method of generating different views of a 3D object, from very sparse input views. In this repo, I'll be implementing it myself. 

### The major things I'll be making are:

The MLP network that takes in any point in 3D space, along with the direction the camera is facing, to give the volume density at that point, and the RGB color at that point

The ray generator and ray marcher, which will create a ray from the camera for every single pixel of the output render, and break the ray into small segments respectively

The volume renderer which will use these rays, along with some simple equations to render the output equation. The volume renderer will be completely
differentiable, meaning I can run a loss function on the generated image, and the actual image for this view.

I will train all these on the Lego Bulldozer Tiny dataset, which is what the original NeRF paper trained on.

After this, I will script my own blender add-on. The user will set a camera position and angle, and hit a button. The NeRF will then render the scene for them
