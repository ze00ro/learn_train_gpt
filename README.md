自己练手用的，主要是为了学习 llm 及其周边，

然而资源有限，所以选择小的 decoder 模型 - GPT2 进行模拟。

没特殊说明的话，都在 单卡 16G T4 上玩的。

通过触发 OOM，然后修改参数或使用工具优化来学习。

因为预处理的时候都处理成了最大长度，所以训练时占用比较稳定。


## simple huggingface trainer

| batch_size | gpu memory | time      | loss            |
|------------|------------|-----------|-----------------|
| 1          | 4606MB     | 535.2154s | 2.1341215042840 |
| 2          | 7514MB     | 495.3674s | 2.2626221429734 |
| 4          | 12936MB    | 479.8573s | 2.3847114199683 |
| 8          | OOM        | -         | -               |

## add fp16

| batch_size | gpu memory | time      | loss            |
|------------|------------|-----------|-----------------|
| 1          | 4328MB     | 535.2154s | 2.1341215042840 |
| 2          | 6470MB     | 277.2177 | 2.2659505208333335 |
| 4          | 10652MB    | 253.3446s | 2.389812215169271 |
| 8          | OOM        | -         | -               |
