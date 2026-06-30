# reUI Agent Entry Point

本项目是科研可审计原型，不是通用农机自动驾驶产品。任何 AI 或开发者修改代码前必须依次阅读：

1. `AI_RULES.md`
2. `docs/DECISIONS.md`
3. `docs/DATA_CONTRACTS.md`
4. 与修改模块相关的 `DECISION-*` 代码锚点和不变量测试

修改前必须列出受影响的 `DECISION-*` 编号。修改科研链路时，不得只改实现而不核对对应测试。

默认验证环境：

```powershell
D:\zhl\anaconda\envs\game\python.exe -m compileall -q .
D:\zhl\anaconda\envs\game\python.exe -m pytest -q
```

本文件只负责入口。完整规则以 `AI_RULES.md` 和 `docs/DECISIONS.md` 为准。
