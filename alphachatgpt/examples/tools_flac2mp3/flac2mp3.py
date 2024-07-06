# from ncmdump import dump
import os,fnmatch

print("软件仅供学习交流，请勿用于商业及非法用途，如产生法律纠纷与本人无关。")
print("------")
print("请在下方输入网易云音乐下载路径，请确保输入正确，否则无法正常转换。")
print("如果您不知道路径在哪里，在网易云客户端中点击：设置 --> 下载设置，即可看到下载路径。")
print("如留空，默认: C:\\CloudMusic\\")
download_folder = input("下载路径：") or "C:\\CloudMusic\\"
os.system('cls')
waiting = True
print("当前下载路径：" + download_folder)
print("您现在可以在网易云音乐客户端中直接下载歌曲，本工具会自动将 flac 转换成 mp3 格式。")
print("等待转换...")



def flac_to_mp3(input_file, output_file):
    """
    使用ffmpeg将FLAC文件转换为MP3文件
    :param input_file: 输入的FLAC文件路径
    :param output_file: 输出的MP3文件路径

    # 示例用法
    input_flac = "path/to/your/input/file.flac"
    output_mp3 = "path/to/your/output/file.mp3"

    flac_to_mp3(input_flac, output_mp3)
    """
    import subprocess
    
    # FFmpeg命令，包括比特率等参数可根据需要调整
    cmd = f"ffmpeg -i \"{input_file}\" -vn -ar 44100 -ab 192k -f mp3 \"{output_file}\""
    
    try:
        # 使用subprocess.run执行命令，capture_output=True可以捕获输出信息，text=True使输出为文本形式
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"转换完成，文件已保存为：{output_file}")
    except subprocess.CalledProcessError as e:
        print(f"转换过程中发生错误：{e.stderr}")

from moviepy.editor import AudioFileClip

def flac_to_mp3_moviepy(input_file, output_file):
    """
    使用moviepy将FLAC文件转换为MP3文件
    :param input_file: 输入的FLAC文件路径
    :param output_file: 输出的MP3文件路径
    
    pip install moviepy
    # 示例用法
    input_flac = "path/to/your/input/file.flac"
    output_mp3 = "path/to/your/output/file.mp3"

    flac_to_mp3_moviepy(input_flac, output_mp3)

    """
    # 加载FLAC文件
    audio_clip = AudioFileClip(input_file)
    
    # 导出为MP3
    audio_clip.write_audiofile(output_file, codec='mp3')
    
    # 关闭音频 clip 以释放资源
    audio_clip.close()
    
    print(f"转换完成，文件已保存为：{output_file}")


def all_files(root, patterns='*', single_level=False, yield_folder=False):
    patterns = patterns.split(';')
    for path, subdirs, files in os.walk(root):
        if yield_folder:
            files.extend(subdirs)
        files.sort()
        for fname in files:
            for pt in patterns:
                if fnmatch.fnmatch(fname, pt):
                    yield os.path.join(path, fname)
                    break
        if single_level:
            break

def run():
    global waiting
    thefile=list(all_files(download_folder, '*.flac'))
    for item in thefile:
        if(waiting == True):
            waiting = False
            os.system('cls')
            print(thefile)
        new_item = str(item).replace(".flac", ".mp3")
        print(flac_to_mp3_moviepy(item, new_item),"转换成功！")
        # delete = os.remove(item)
        
    print("转换完成！！！")

run()