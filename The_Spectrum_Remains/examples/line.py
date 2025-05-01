import numpy as np
import matplotlib.pyplot as plt

# Image dimensions
height = 300
width = 50

# Create vertical gradient: 1.0 (white) at top to 0.0 (black) at bottom
intensity = np.linspace(1.0, 0.0, height).reshape(-1, 1)
intensity = np.repeat(intensity, width, axis=1)

# Create grayscale RGB image
rgb_image = np.stack([intensity]*3, axis=2)

# Plot the image
plt.figure(figsize=(2, 6))
plt.imshow(rgb_image, aspect='auto')
plt.axis('off')
plt.title("White to Black Intensity", fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig('intensity_bar.png', dpi=300, bbox_inches='tight')