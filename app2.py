import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from PIL import Image
#import matplotlib.pyplot as plt
import io

def main_network(image):
    class UNetGenerator(torch.nn.Module):
        def __init__(self, in_channels=1, out_channels=3):
            super(UNetGenerator, self).__init__()

            def down_block(in_channels, out_channels, normalize=True):
                layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
                if normalize:
                    layers.append(torch.nn.BatchNorm2d(out_channels))
                layers.append(torch.nn.LeakyReLU(0.2))
                return torch.nn.Sequential(*layers)

            def up_block(in_channels, out_channels, dropout=0.0):
                layers = [torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU(inplace=True)]
                if dropout:
                    layers.append(torch.nn.Dropout(dropout))
                return torch.nn.Sequential(*layers)

            self.down1 = down_block(in_channels, 64, normalize=False)
            self.down2 = down_block(64, 128)
            self.down3 = down_block(128, 256)
            self.down4 = down_block(256, 512)
            self.down5 = down_block(512, 512)
            self.down6 = down_block(512, 512)
            self.down7 = down_block(512, 512)
            self.down8 = down_block(512, 512, normalize=False)

            self.up1 = up_block(512, 512, dropout=0.5)
            self.up2 = up_block(1024, 512, dropout=0.5)
            self.up3 = up_block(1024, 512, dropout=0.5)
            self.up4 = up_block(1024, 512)
            self.up5 = up_block(1024, 256)
            self.up6 = up_block(512, 128)
            self.up7 = up_block(256, 64)
            self.up8 = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                torch.nn.Tanh()
            )

        def forward(self, x):
            d1 = self.down1(x)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            d7 = self.down7(d6)
            d8 = self.down8(d7)

            u1 = self.up1(d8)
            u2 = self.up2(torch.cat([u1, d7], 1))
            u3 = self.up3(torch.cat([u2, d6], 1))
            u4 = self.up4(torch.cat([u3, d5], 1))
            u5 = self.up5(torch.cat([u4, d4], 1))
            u6 = self.up6(torch.cat([u5, d3], 1))
            u7 = self.up7(torch.cat([u6, d2], 1))
            u8 = self.up8(torch.cat([u7, d1], 1))

            return u8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = UNetGenerator(in_channels=1, out_channels=3).to(device)
    generator.load_state_dict(torch.load('generator_epoch.pth', map_location=device))

    generator.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale image
    ])

    def transform_image(image):
        input_img = transform(image)
        input_img = input_img.unsqueeze(0).to(device)  # Add batch dimension

        # Generate the colorized image
        with torch.no_grad():
            fake_img = generator(input_img)
            fake_img = (fake_img * 0.5) + 0.5  # Rescale to [0, 1]
            fake_img = fake_img.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC format

        # Convert to PIL Image
        fake_img = (fake_img * 255).astype("uint8")
        return Image.fromarray(fake_img)
    out_= transform_image(image)
    return out_

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file into an image
    image1 = Image.open(uploaded_file)  
    
    # Convert the image to grayscale
    grayscale_image = image1.convert('L')
    
    # Display the grayscale image
    st.image(grayscale_image, caption='Grayscale Image', use_column_width=True)
    
    # Apply the neural network (or any other processing) only after the image is uploaded
    with st.spinner('Processing image...'):
        colorized_image = main_network(grayscale_image)
        st.image(colorized_image, caption='Colorized Image', use_column_width=True)
        buf = io.BytesIO()
        colorized_image.save(buf, format='PNG')
        buf.seek(0)
        
        # Provide a download button
        st.download_button(
            label="Download Colorized Image",
            data=buf,
            file_name="colorized_image.png",
            mime="image/png"
        )
else:
    st.write("Please upload an image.")
