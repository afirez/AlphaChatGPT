import numpy as np
import matplotlib.pylab as plt

import torch
from torch.nn import functional as F

"""
lm
attention multi Head Block

multi layer:
  1. 简单线性问题，使用简单感知机即可；
  2. 简单非线性问题， 使用一层隐藏层即可
  3. 复杂非线性问题， 使用深度神经网络
"""

text = '''
Cats, cats, everywhere
Furry balls without a care
Purring, meowing, licking fur
Hunting mice, they always purr

Cats, cats, on the prowl
Jumping high, never a scowl
Whiskers twitching, eyes alert
Tail in air, ready to assert

Cats, cats, so much fun
Cuddling close in the sun
Stretching out, napping long
Playing with string, never wrong

Cats, cats, always cool
Lapping milk, acting like a fool
Mysterious, charming, full of grace
Cats are simply ace

Cats, cats, with silky fur
Making biscuits, they always purr
Sitting high, looking down
Claiming everything as their crown

Cats, cats, with eyes so bright
Chasing shadows, day or night
Curled up warm, on your lap
Purring gently, taking a nap

Cats, cats, with playful paws
Hiding, stalking, never pause
Jumping, leaping, so agile
Graceful creatures, never fragile

Cats, cats, our feline friends
Bringing joy that never ends
Loving us, without a doubt
Cats are what life's all about

Cats, cats, everywhere I see
Furry creatures, cute as can be
Rubbing against our legs
Asking for treats, without begs

Cats, cats, with their regal stance
Graceful movements, they enhance
But we love them all the same
Our little friends, never tame

Cats, cats, so full of love
Watching over us from above
Protecting us from any harm
Always there, with their charm

Cats, cats, with their curious ways
Exploring nooks, and hiding in bays
Living life with style and grace
Cats are always in first place

Cats, cats, so full of fun
Chasing toys, never done
Hiding in boxes, or paper bags
Making us laugh, never drags

Cats, cats, with their own minds
Sitting in the sun, never blinds
Chasing strings, and balls of yarn
They never tire, oh what a charm

Cats, cats, with calming purrs
Cuddling close, to be yours
Giving love, without any fuss
Their presence, a comfort to us

Cats, cats, always at ease
Living life, as they please
Bringing joy, to all they meet
Cats, our furry friends, so sweet

Cats, cats, with eyes so bright
Guiding us through the darkest night
Purring softly, by our side
Comforting us, as we hide

Cats, cats, with softest fur
Nuzzling close, making a purr
In our lap, they take a rest
We're lucky to have, such a guest

Cats, cats, with their playful ways
Entertaining us, on the laziest days
Chasing shadows, or a feather
Making us smile, always together

Cats, cats, with hearts so pure
Bringing love, that will endure
Their presence, a blessing indeed
Cats, our friends, we shall never need

Cats, cats, with their little quirks
Scratching posts, and tiny perks
Licking paws, cleaning their face
Chasing tails, all over the place

Cats, cats, with their playful hearts
Chasing toys, and little carts
Their antics, bringing us joy
Cats, our little angels, oh so coy

Cats, cats, with their gentle souls
Lifting spirits, making us whole
In their eyes, we see the light
Bringing peace, that feels so right

Cats, cats, with their gentle purr
Calming us, when we're feeling a stir
Snuggling close, to keep us warm
Cats, our little cuddle storm

Cats, cats, with their playful heart
Jumping high, right from the start
Bouncing around, like little springs
Cats, our little entertainers, with wings

Cats, cats, with their loving grace
Their soft purrs, caress our face
In their embrace, we feel at peace
Cats, our little comfort, never to cease

Cats, cats, with their loving ways
Cuddling close, on the darkest days
In the garden, or up in a tree
Cats, our little explorers, always free

'''

text = text.lower()
chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
data = [stoi[c] for c in text]
vocab_size = len(chars)

ins = 64 # 16 ， 32, 64 输入长度太小可能不足以对句子建模
outs = vocab_size
nodes = 320 # 200 

lr = 0.003

n_emb = 120 # 32
embed = torch.randn(vocab_size, n_emb)
pos = torch.randn(ins, n_emb)

data = torch.tensor(data).long()
# xs = torch.stack([ data[i:i+ins] for i in range(len(data) - ins) ] )
# xy = torch.stack([ data[i+ins:i+ins+1] for i in range(len(data) - ins) ] )

params = []

def weight(ins, outs):
    ws = torch.randn(ins, outs) * 0.1
    ws = ws.requires_grad_(True)
    return ws

class Head():
    def __init__(self) -> None:
        self.wv = weight(n_emb, n_emb//4)
        # self.wq = weight(n_emb, n_emb//4)
        # self.wk = weight(n_emb, n_emb//4)
        self.wr = weight(n_emb, ins)

        params.append(self.wv)
        # params.append(self.wq)
        # params.append(self.wk)
        params.append(self.wr)

    def forward(self, x):
        v = x @ self.wv
        # q = x @ self.wq
        # v = x @ self.wk
        
        # attn = (q @ k.transpose(-2, -1)) / k.shape[0] ** 0.5

        re_weight = x @ self.wr
        
        tril = torch.tril(re_weight)
        tril = tril.masked_fill(tril == 0, -1e10)
        rew = F.softmax(tril, dim=-1)
        x = rew @ v

        return x

class Block():
    def __init__(self) -> None:
        self.heads = [Head(), Head(), Head(), Head()]
        self.w0 = weight(n_emb, nodes)
        self.w1 = weight(nodes, n_emb)

        params.append(self.w0)
        params.append(self.w1)
    
    def forward(self, x):
        x = torch.cat([head.forward(x) for head in self.heads], dim=-1)

        x = torch.relu(x @ self.w0)
        x = torch.relu(x @ self.w1)

        return x



class Model():
    def __init__(self) -> None:
        self.blocks = [Block(), Block(), Block()]
        self.w2 = weight(n_emb, outs)

        params.append(self.w2)

    def forward(self, x):
        x = embed[x] + pos

        x = x + self.blocks[0].forward(x)
        x = x + self.blocks[1].forward(x)
        x = x + self.blocks[2].forward(x)

        yh = (x @ self.w2) 
        return yh

model =  Model()

# optimiser = torch.optim.SGD(params, lr)
optimiser = torch.optim.Adam(params, lr)
print("params:", sum(p.numel() for p in params))

errs = []
for i in range(5000):

    b = torch.randint(len(data)-ins, (100,))
    xs = torch.stack([data[i:i+ins] for i in b])
    # ys = torch.stack([data[i+ins:i+ins+1] for i in b])
    ys = torch.stack([data[i+1:i+ins+1] for i in b])

    yh = model.forward(xs.float())

    # loss = F.mse_loss(yh, ys)
    loss = F.cross_entropy(yh.view(-1, vocab_size), ys.long().view(-1)) 
    optimiser.zero_grad()

    loss.backward()
    optimiser.step()

    e = loss.item()
    if i % 500 == 0:
        print(f"loss: {e}")

    errs.append(e)

plt.figure(1)
plt.plot(errs)

plt.figure(2)
plt.plot(ys)
plt.plot(torch.argmax(yh.detach(), dim=-1))

plt.show()

def generate_text(s, length):
    print("generated text:")
    print("---------------------")
    gen_text = ""
    for i in range(length):
        yh = model.forward(s)
        prob = F.softmax(yh[-1, :]*1, dim=0)
        # pred = torch.argmax(yh).item()
        pred = torch.multinomial(prob, num_samples=1).item()        
        s = torch.roll(s, -1)
        s[-1] = pred
        gen_text += itos[pred]
    
    print(gen_text)

generate_text(xs[0], 3000)