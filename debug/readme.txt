# 预训练数据集
bookcorpus  
https://huggingface.co/datasets/bookcorpus

wikipedia
https://huggingface.co/datasets/wikipedia


#apex 安装需要参看官网，不能直接用pip安装，不要用主分支，
https://github.com/NVIDIA/apex
要下载 22.04-dev 分支的zip包。然后参考这个分支的readme文档进行安装


GLMForConditionalGeneration --> 任务入口
    1 GLMModel  ---> glm
       1 VocabEmbedding    ---> word embedding 
       2 GLMStack   ---> transformer layer
           1 Embedding   ---> position embedding
           2 Embedding   ---> block_position_embedding
           3 GLMBlock (xN)  ---> 
                1 LayerNorm
                2 SelfAttention
                3 LayerNorm
                4 MLP
           4 LayerNorm   ---> 
       3 Linear ---> ? weight 来自 word embedding