## 安装

配置完成 [LangBot](https://github.com/RockChinQ/QChatGPT) 主程序后使用管理员账号向机器人发送命令即可安装：

```
!plugin get <插件发布仓库地址>
```
或查看详细的[插件安装说明](https://github.com/RockChinQ/QChatGPT/wiki/5-%E6%8F%92%E4%BB%B6%E4%BD%BF%E7%94%A8)

## 使用  
1、需要先部署[RVC变声器项目](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)，这里不做介绍  

2、将"rvc_fastapi.py"和"开启接口服务.bat"文件放到RVC项目的主目录下，使用前需要先双击打开"开启接口服务.bat"文件  

3、需要下载安装ffmpeg   

4、前往https://www.alapi.cn/  进行注册（截至上传时是免费的）

   在token管理中点击Copy复制Token

   在本插件文件夹下main.py文件中找到这行，并替换成你获取到的token（不要弄丢引号）

```
token = 'YOURTOKEN'  # 请将这里的'YOUR_TOKEN'替换为你实际获取的token
cookie = "YOUR_COOKIE"  # 请将这里的'YOUR_COOKIE'替换为你实际获取的cookie
RVC_logs_path = r"F:\RVC\RVC1006Nvidia\logs"  # 请将这里的"F:\RVC\RVC1006Nvidia\logs"替换为你部署的RVC项目的logs文件夹的路径
```

另外main.py中的cookie为你网易云音乐的cookie，如果没有请将其设为空，若设为空一些vip歌曲可能只能获取30秒  

只能获取网易云音乐上有的音乐  

5、tmp文件夹下为UVR的模型文件，文件大小较大，下载较慢请耐心等待，在首次使用时也会自动下载，但需要魔法  

## 注意

1、请检查RVC项目中"\assets\weights"中的模型文件名称xxx.pth和".\logs\xxx\added_abcdefg1234.index"中xxx的名称相同且index文件必须放在".\logs\xxx\"目录下  

2、变声前需要用UVR模型对音频去伴奏和混响，依赖电脑性能，速度过慢或爆显存可以适当修改main.py中的batch_size大小，本人3070 8G显存 现在的batch_size大小刚好合适，大家可以做参考去调节。  

   去伴奏过程慢调节new_mdx_params中的batch_size。去和声或去混响过程慢调节new_vr_params中的batch_size
```
new_mdx_params = {"hop_length": 1024, "segment_size": 256, "overlap": 8, "batch_size": 2, "enable_denoise": False}
new_vr_params = {"batch_size": 2, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False}
```

## 配置GPT

向bot发送：
```
#学习翻唱[music_name][number][model_name]
```
music_name 为需要翻唱的歌曲名称，可以限制歌手，但必须为网易云音乐中的歌曲  

number 为音调的变化，如果原歌曲为男声，模型为女声，则number为12，升调；如果原歌曲为女声，模型为男声，则number为-12，降调；如果是男变男或女变女则number为0。注意12和-12只是参考，可以微调。  

model_name 为RVC的模型名称，即 "xxx.pth和logs\xxx\.index文件" 中的xxx  

注意：#学习翻唱和三个'[]'不可缺少    

整个过程需要几分钟请耐心等待，实测下来3070 8G显存完整处理一首歌约2、3分钟   

![示例图片](https://github.com/zzseki/LangBot_RVC_Music/blob/main/tmp/F3595CB91817E2991F59C356AD573638.jpg)
