import os
import subprocess

if __name__ == "__main__":
    image_directory = "figure"

    output_mp4 = "skyplot.mp4"

    with open("input.txt", "w") as file:
        for image in sorted(os.listdir(image_directory)):
            if image.startswith("skyplot_") and image.endswith(".png"):
                file.write(f"file '{os.path.join(image_directory, image)}'\n")
                file.write("duration 0.1\n")

    ffmpeg_command = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "input.txt",
        "-vf",
        "scale=640:-1",
        "-y",
        output_mp4,
    ]

    subprocess.run(ffmpeg_command)
    os.remove("input.txt")
