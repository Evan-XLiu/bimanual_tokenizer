# Codex / codex88 配置审计与本地复现

> 审计时间：2026-09-14  
> 审计环境：Linux，Codex CLI `0.153.0`  
> 安全说明：本文不包含任何现有 access token、refresh token 或 API key。

## 结论

当前机器并没有安装两个不同的 Codex 程序。`codex` 和 `codex88` 共用同一个
Codex CLI 二进制，区别仅在于启动时使用了不同的 `CODEX_HOME`：

```text
codex
  -> ~/.local/bin/codex
  -> ~/.codex/packages/standalone/current/bin/codex
  -> 默认读取 ~/.codex/

codex88
  -> alias: CODEX_HOME="$HOME/.codex-88apis" codex
  -> 仍调用同一个 ~/.local/bin/codex
  -> 改为读取 ~/.codex-88apis/
```

这正是两套配置互不干扰的关键。OpenAI 官方文档说明，`CODEX_HOME` 默认是
`~/.codex`，它控制 Codex 的配置、认证、日志、会话、skills 和安装元数据所在的
根目录；显式设置时，该目录必须已经存在。

## 当前机器上的实际配置

### CLI 安装

- 可见命令：`~/.local/bin/codex`
- 实际版本目录：`~/.codex/packages/standalone/releases/0.153.0-x86_64-unknown-linux-musl/`
- `~/.codex/packages/standalone/current` 指向上述版本目录
- `codex` 与 `codex88` 显示的版本均为 `codex-cli 0.153.0`

这符合官方 standalone installer 的目录布局。不要为 `codex88` 再安装一份 CLI。

### 普通 codex

配置根目录为 `~/.codex`，认证方式为 ChatGPT 登录。当前的核心
`~/.codex/config.toml` 是：

```toml
model = "gpt-5.6-sol"
model_reasoning_effort = "medium"
```

此外，原文件记录了若干仅适用于服务器的受信任项目绝对路径，以及一个 TUI
提示状态。迁移到个人电脑时不应该照抄这些服务器路径；在本机实际项目中完成
信任确认即可。

`~/.codex/auth.json` 中保存的是 ChatGPT 登录状态和令牌。该文件权限为 `0600`，
不应复制到另一台电脑，也不应提交到 Git；应在新电脑上重新运行登录流程。

### codex88

`~/.bashrc` 中的实际定义为：

```bash
# Separate Codex configuration for the 88apis provider.
alias codex88='CODEX_HOME="$HOME/.codex-88apis" codex'
```

它读取 `~/.codex-88apis/config.toml`：

```toml
model = "gpt-5.6-sol"
model_provider = "custom"
model_reasoning_effort = "medium"
cli_auth_credentials_store = "file"

[model_providers.custom]
name = "88apis"
base_url = "https://88apis.top/v1"
wire_api = "responses"
requires_openai_auth = true
```

该目录使用 API key 认证。`auth.json` 只包含 `auth_mode` 和
`OPENAI_API_KEY` 两类字段，文件权限为 `0600`。本文不会记录或迁移实际 key。

和主配置一样，原文件还有若干服务器项目的 `trust_level = "trusted"`；本地复现时
应删除或替换为本地项目路径。

## 在本地复现

以下命令适用于 macOS/Linux；Windows 见后文。

### 1. 安装 Codex CLI

使用官方 standalone installer：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

确认安装结果：

```bash
command -v codex
codex --version
```

如果 shell 找不到命令，确认 `~/.local/bin` 在 `PATH` 中：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

然后把这一行加入 `~/.bashrc` 或 `~/.zshrc`。

### 2. 配置普通 codex

创建默认配置目录：

```bash
mkdir -p "$HOME/.codex"
chmod 700 "$HOME/.codex"
```

创建 `~/.codex/config.toml`：

```toml
model = "gpt-5.6-sol"
model_reasoning_effort = "medium"
```

然后登录：

```bash
codex login
```

根据终端提示选择 ChatGPT 登录。模型是否可用取决于本地登录账号和工作区权限；
如果 `gpt-5.6-sol` 不可用，可启动 Codex 后用 `/model` 选择账号实际可用的模型，
再更新 `config.toml`。

### 3. 创建独立的 codex88 配置

```bash
mkdir -p "$HOME/.codex-88apis"
chmod 700 "$HOME/.codex-88apis"
```

创建 `~/.codex-88apis/config.toml`，内容如下：

```toml
model = "gpt-5.6-sol"
model_provider = "custom"
model_reasoning_effort = "medium"
cli_auth_credentials_store = "file"

[model_providers.custom]
name = "88apis"
base_url = "https://88apis.top/v1"
wire_api = "responses"
requires_openai_auth = true
```

为这套配置单独登录。先从 88apis 服务方获取自己的 API key，然后执行：

```bash
read -rsp "88apis API key: " CODEX88_API_KEY; printf '\n'
printf '%s' "$CODEX88_API_KEY" | \
  CODEX_HOME="$HOME/.codex-88apis" codex login --with-api-key
unset CODEX88_API_KEY
chmod 600 "$HOME/.codex-88apis/auth.json"
```

不要把真实 key 直接写入 shell 启动文件、`config.toml`、命令历史或本文档。

### 4. 添加 codex88 命令

Bash 用户把下面内容加入 `~/.bashrc`，Zsh 用户加入 `~/.zshrc`：

```bash
alias codex88='CODEX_HOME="$HOME/.codex-88apis" codex'
```

重新载入配置：

```bash
source "$HOME/.bashrc"   # Zsh 则使用 source "$HOME/.zshrc"
```

需要注意，alias 通常只在交互 shell 中展开。脚本、CI 或远程非交互命令中应直接写：

```bash
CODEX_HOME="$HOME/.codex-88apis" codex exec "你的任务"
```

### 5. 验证两套配置

```bash
type -a codex
type -a codex88

codex --version
codex login status

codex88 --version
CODEX_HOME="$HOME/.codex-88apis" codex login status
```

预期结果：

- 两条命令的 CLI 版本相同；
- `codex login status` 显示 ChatGPT 登录；
- 第二套状态显示 API key 登录；
- 在两个客户端中运行 `/status` 时，provider/config root 应不同；
- 两边的会话历史和认证状态互不覆盖。

## Windows PowerShell 复现

安装：

```powershell
irm https://chatgpt.com/codex/install.ps1 | iex
```

创建两套目录并写入与上文相同的两个 `config.toml`。Windows 的 home 目录通常通过
`$HOME` 访问，因此路径分别是 `$HOME\.codex` 和 `$HOME\.codex-88apis`。

PowerShell 不使用 Bash alias 语法，可在 `$PROFILE` 中定义函数：

```powershell
function codex88 {
    $oldCodexHome = $env:CODEX_HOME
    try {
        $env:CODEX_HOME = Join-Path $HOME ".codex-88apis"
        codex @args
    }
    finally {
        $env:CODEX_HOME = $oldCodexHome
    }
}
```

登录第二套配置时：

```powershell
$env:CODEX_HOME = Join-Path $HOME ".codex-88apis"
$secureKey = Read-Host "88apis API key" -AsSecureString
$keyPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPtr) | codex login --with-api-key
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPtr)
    Remove-Variable secureKey, keyPtr -ErrorAction SilentlyContinue
    Remove-Item Env:CODEX_HOME -ErrorAction SilentlyContinue
}
```

## 安全与兼容性注意事项

1. `88apis.top` 是第三方 endpoint，不是 OpenAI 官方域名。发送给该 provider 的提示、
   代码上下文和工具相关数据受该服务方的安全与隐私策略约束。使用前应自行确认其
   可信度、数据保留政策和计费方式。
2. 不要从服务器复制 `auth.json`。普通 Codex 的 ChatGPT token 和 codex88 的 API key
   都应在目标电脑重新生成或登录。
3. `cli_auth_credentials_store = "file"` 意味着认证数据存储在文件中。应确保目录权限
   为 `0700`、认证文件权限为 `0600`，并避免同步到公开云盘或 Git。
4. `trust_level` 应逐个本地项目确认。不要把 `/` 或整个 home 目录无条件设为 trusted。
5. 两套配置都指定了 `gpt-5.6-sol`。第三方 provider 是否真正支持该模型名和 Responses
   协议，需要以该服务方当前说明和实际请求结果为准。
6. 官方说明 `CODEX_HOME` 必须预先存在，因此应先 `mkdir`，再运行 `codex88`。

## 官方参考

- [Codex CLI：安装与登录](https://learn.chatgpt.com/docs/codex/cli)
- [Codex Config basics](https://learn.chatgpt.com/docs/config-file/config-basic)
- [Codex 环境变量（含 CODEX_HOME）](https://learn.chatgpt.com/docs/config-file/environment-variables)
- [Codex 配置字段参考](https://learn.chatgpt.com/docs/config-file/config-reference)
- [Codex 示例配置（含 custom provider）](https://learn.chatgpt.com/docs/config-file/config-sample)

