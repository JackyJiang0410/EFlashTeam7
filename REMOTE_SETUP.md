# Remote Server Setup Guide for Cursor

## 连接到远程服务器 (Connect to Remote Server)

### 1. SSH 连接信息 (SSH Connection Info)

根据您的终端，远程服务器信息：
- **Host**: `ece-a52003`
- **User**: `hj8947@austin.utexas.edu`
- **完整地址**: `hj8947@ece-a52003.austin.utexas.edu` 或类似格式

### 2. 在 Cursor 中设置 Remote-SSH

#### 步骤 A: 安装扩展
1. 打开 Cursor
2. 按 `Cmd+Shift+X` 打开扩展市场
3. 搜索 "Remote - SSH"
4. 点击安装

#### 步骤 B: 添加 SSH 主机
1. 按 `Cmd+Shift+P` 打开命令面板
2. 输入: `Remote-SSH: Connect to Host...`
3. 选择 `+ Add New SSH Host...`
4. 输入 SSH 命令（尝试以下格式之一）:
   ```
   ssh hj8947@ece-a52003.austin.utexas.edu
   ```
   或
   ```
   ssh hj8947@austin.utexas.edu@ece-a52003
   ```
5. 选择保存配置文件的位置（通常选择第一个：`~/.ssh/config`）

#### 步骤 C: 编辑 SSH 配置（可选）

如果需要自定义配置，编辑 `~/.ssh/config`:
```ssh
Host ece-a52003
    HostName ece-a52003.austin.utexas.edu
    User hj8947@austin.utexas.edu
    # 如果需要指定端口:
    # Port 22
    # 如果使用密钥文件:
    # IdentityFile ~/.ssh/id_rsa
```

#### 步骤 D: 连接到服务器
1. 再次按 `Cmd+Shift+P`
2. 选择 `Remote-SSH: Connect to Host...`
3. 选择 `ece-a52003`（或您配置的主机名）
4. 选择操作系统：**Linux**
5. 输入密码（或使用 SSH 密钥认证）

### 3. 打开远程项目文件夹

连接成功后：
1. 点击 **"Open Folder"** 按钮
2. 输入项目路径，例如：
   ```
   ~/eFlesh
   ```
   或完整路径：
   ```
   /home/hj8947@austin.utexas.edu/eFlesh
   ```

### 4. 在远程服务器上运行训练

连接并打开项目后，在 Cursor 的终端中运行：

```bash
# 确保在项目目录
cd ~/eFlesh  # 或您的项目路径

# 激活环境（如果使用 conda）
conda activate eflesh

# 或如果使用 venv
source venv/bin/activate

# 运行训练
python characterization/train.py \
    --folder Data/local_sin_3*3/ \
    --epochs 10 \
    --batch_size 32
```

---

## 常见问题 (Common Issues)

### 问题 1: "Host key verification failed"
**解决**: 在本地终端运行：
```bash
ssh-keyscan ece-a52003.austin.utexas.edu >> ~/.ssh/known_hosts
```

### 问题 2: "Permission denied"
**解决**: 
- 检查用户名是否正确
- 确认 SSH 密钥已设置，或准备好密码
- 联系服务器管理员确认访问权限

### 问题 3: "Connection timeout"
**解决**:
- 确认您在学校 VPN 上（如果需要）
- 检查防火墙设置
- 尝试使用完整域名而不是主机名

### 问题 4: 找不到文件路径
**解决**: 在远程服务器上确认项目位置：
```bash
# 在远程终端运行
pwd
ls -la
find ~ -name "train.py" -type f 2>/dev/null
```

---

## 快速检查清单 (Quick Checklist)

- [ ] Remote-SSH 扩展已安装
- [ ] SSH 主机已添加到配置
- [ ] 成功连接到远程服务器
- [ ] 项目文件夹已打开
- [ ] Python 环境已激活
- [ ] 训练脚本可以运行

---

## 替代方案 (Alternative)

如果 Remote-SSH 有问题，您可以：

1. **使用本地编辑 + 远程执行**:
   - 在本地 Cursor 编辑代码
   - 使用 `scp` 同步文件到服务器
   - 在服务器终端运行训练

2. **使用 Jupyter Notebook**:
   - 在服务器上启动 Jupyter
   - 在本地浏览器访问

3. **使用 tmux/screen**:
   - SSH 到服务器
   - 启动 tmux session
   - 在 session 中运行长时间训练任务

---

## 需要帮助？

如果遇到问题，提供以下信息：
- SSH 连接命令（您在终端中使用的）
- 错误消息截图
- 服务器操作系统版本（`uname -a`）
- Python 版本（`python3 --version`）


