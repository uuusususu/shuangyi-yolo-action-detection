@echo off
chcp 65001 >nul
setlocal
echo 正在请求管理员权限以放行训练台内网访问端口（TCP 8771）...
net session >nul 2>&1
if not %errorlevel% == 0 (
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
netsh advfirewall firewall delete rule name="LSTM_Workbench_8771" >nul 2>&1
netsh advfirewall firewall add rule name="LSTM_Workbench_8771" dir=in action=allow protocol=TCP localport=8771 profile=private,domain,public
if %errorlevel% == 0 (
  echo.
  echo [成功] 已放行 TCP 8771 入站端口！现在内网/网线直连电脑可直接打开训练台内网网址。
) else (
  echo.
  echo [失败] 未能添加防火墙规则，请右键选择“以管理员身份运行”。
)
pause
endlocal
