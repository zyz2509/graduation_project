x = torch.randn(8*2048, 3).cuda()
y = torch.randn(8*2048, 64).cuda()
z = [2048 for _ in range(8)]
net = SAModule(64, 0.4, 128).cuda()
net1 = SAModule(128, 1, 256).cuda()
net2 = SAModule(256, 1.5, 256).cuda()
for _ in range(10):
    s = time.time()
    a, b, c = net(x, y, z)
    a, b, c = net1(a, b, c)
    a, b, c = net2(a, b, c)
    ss = time.time()
    print(c)
    print(ss-s)