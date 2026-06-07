# Comm Architecture · 通信架构

## 模块定位

本模块梳理多模块机器人系统中**低延迟、高可靠**的通信方案，覆盖从进程内到跨设备的全栈传输设计。

## 核心主题

| 主题 | 说明 |
|------|------|
| UDP | 实时音视频、传感器流；丢包容忍与重传策略 |
| TCP | 可靠控制指令、文件传输、状态同步 |
| WebSocket | 浏览器/移动端与后端的双向长连接 |
| ROS 多模块 | Topic/Service/Action 选型、节点拓扑与 QoS 配置 |

## 典型问题

- 语音流用 UDP 还是 WebSocket？延迟与可靠性的 trade-off 如何量化？
- ROS2 与自研 UDP 协议如何共存、边界如何划分？
- 局域网内多机器人/多客户端的 discovery 与负载均衡？

## 目录建议

```
Comm_Architecture/
├── protocols/        # 各协议对比与选型决策
├── ros/              # ROS1/ROS2 节点设计与 launch
├── websocket/        # WS 网关、心跳、重连
├── udp_tcp/          # 自定义二进制/Protobuf 协议
└── latency_tests/    # 端到端延迟与带宽测试
```

## 关联模块

- `Voice_Interaction` — 语音与控制流的传输载体
- `Edge_Deployment` — 局域网私有化部署下的网络拓扑
- `Ops_Automation` — 网络排查与防火墙/端口配置
