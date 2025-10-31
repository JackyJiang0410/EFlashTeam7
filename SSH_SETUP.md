# SSH 密钥设置指南 (SSH Key Setup Guide)

## 问题 (Problem)
Cursor Remote-SSH 无法处理 Duo 两步验证，导致连接失败。

## 解决方案 (Solution)
设置 SSH 密钥认证，避免每次输入密码和 Duo。

---

## 步骤 1: 密钥已生成 ✅

您的 SSH 公钥已生成：
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIK8y4cQx1j9x9nTW5GgWOzMiND2kAbzTqM4U4bM3YMZk haojiang@ece.utexas.edu
```

---

## 步骤 2: 复制密钥到服务器

### 方法 A: 使用 ssh-copy-id (推荐)

在**终端**中运行（这会触发 Duo 验证一次）：

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub hj8947@ece-a52003.ece.utexas.edu
```

或者如果用户名不同：
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub haojiang@ece-a52003.ece.utexas.edu
```

**注意**: 这一步会要求输入密码和 Duo 验证，但只需做一次。

---

### 方法 B: 手动复制（如果 ssh-copy-id 不可用）

1. **在终端 SSH 到服务器**（会触发 Duo）:
   ```bash
   ssh hj8947@ece-a52003.ece.utexas.edu
   ```

2. **在服务器上创建 .ssh 目录**:
   ```bash
   mkdir -p ~/.ssh
   chmod 700 ~/.ssh
   ```

3. **复制公钥内容**:
   
   在**本地 Mac**，复制以下内容：
   ```
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIK8y4cQx1j9x9nTW5GgWOzMiND2kAbzTqM4U4bM3YMZk haojiang@ece.utexas.edu
   ```

   然后在**服务器**上运行：
   ```bash
   echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIK8y4cQx1j9x9nTW5GgWOzMiND2kAbzTqM4U4bM3YMZk haojiang@ece.utexas.edu" >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   ```

4. **退出服务器**:
   ```bash
   exit
   ```

---

## 步骤 3: 配置 SSH Config

编辑 `~/.ssh/config` 文件，添加服务器配置：

```bash
nano ~/.ssh/config
```

添加以下内容：

```ssh
Host ece-a52003
    HostName ece-a52003.ece.utexas.edu
    User hj8947
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    # 如果需要转发代理
    ForwardAgent yes
```

**注意**: 将 `User hj8947` 改为您的实际用户名（可能是 `haojiang` 或其他）。

保存文件：`Ctrl+O`, `Enter`, `Ctrl+X`

---

## 步骤 4: 测试 SSH 连接

在终端测试（应该不需要密码）:
```bash
ssh ece-a52003
```

如果成功，说明密钥配置正确！

---

## 步骤 5: 在 Cursor 中连接

1. **打开 Cursor**
2. **按 `Cmd+Shift+P`**
3. **选择**: `Remote-SSH: Connect to Host...`
4. **选择**: `ece-a52003`（刚才配置的主机名）

这次应该能成功连接，因为使用的是密钥认证！

---

## 如果还是不行 (Troubleshooting)

### 问题 1: "Permission denied (publickey)"
**解决**: 
- 确认用户名正确
- 检查服务器上的 `~/.ssh/authorized_keys` 文件是否存在且包含您的公钥
- 确认文件权限：`chmod 600 ~/.ssh/authorized_keys`

### 问题 2: 仍然要求密码
**解决**:
- 检查 SSH config 中的 `IdentityFile` 路径是否正确
- 尝试：`ssh -v ece-a52003` 查看详细日志
- 确认密钥权限：`chmod 600 ~/.ssh/id_ed25519`

### 问题 3: Duo 仍然弹出
**解决**:
- 第一次连接可能仍需要 Duo，但后续应该可以跳过
- 如果每次都要求 Duo，联系 ECE-IT: help@ece.utexas.edu

---

## 快速检查清单

- [ ] SSH 密钥已生成 (`~/.ssh/id_ed25519`)
- [ ] 公钥已复制到服务器 (`~/.ssh/authorized_keys`)
- [ ] SSH config 已配置 (`~/.ssh/config`)
- [ ] 终端 SSH 测试成功（无密码）
- [ ] Cursor Remote-SSH 连接成功

---

## 需要帮助？

如果遇到问题：
1. 检查 SSH 详细日志：`ssh -v ece-a52003`
2. 联系 ECE-IT: help@ece.utexas.edu
3. 检查服务器日志：`tail -f /var/log/auth.log`（需要管理员权限）


