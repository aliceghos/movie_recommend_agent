"""HTTP 服务外壳。

``movie_agent/`` 是 agent 领域逻辑，这里只负责把它暴露成带认证、带持久化的
HTTP 服务：路由、鉴权、落库、错误契约。两边的边界是 ``build_agent_runtime()``
与 ``chat_stream()``，服务层不碰 middleware 与工具装配的细节。
"""
