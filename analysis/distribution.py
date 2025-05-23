import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 数据
data = [
    0.6172,
    0.5408,
    0.6516,
    0.6138,
    0.6721,
    0.5906,
    0.6871,
    0.5481,
    0.5512,
    0.5640,
    0.5413,
    0.6787,
    0.5553,
    0.5774,
    0.6992,
    0.6956,
    0.5722,
    0.6857,
    0.6410,
    0.6611,
    0.5709,
    0.6237,
    0.6125,
    0.6701,
    0.5432,
    0.6346,
    0.6316,
    0.5385,
    0.5524,
    0.5816,
    0.6281,
    0.6198,
    0.6452,
    0.5254,
    0.5787,
    0.5392,
    0.5399,
    0.5447,
    0.5247,
    0.6036,
    0.6768,
    0.5526,
    0.6245,
    0.5808,
    0.5886,
    0.6836,
    0.5488,
    0.6565,
    0.5775,
    0.5949,
    0.5971,
    0.6467,
    0.6140,
    0.6615,
    0.6583,
    0.5509,
    0.6223,
    0.6720,
    0.5966,
    0.6782,
    0.6496,
    0.5673,
    0.5867,
    0.6715,
    0.5919,
    0.6453,
    0.6224,
    0.5706,
    0.5651,
    0.6209,
    0.5645,
    0.6089,
    0.5862,
    0.5776,
    0.5920,
    0.6484,
    0.6157,
    0.6530,
    0.5589,
    0.6359,
    0.5345,
    0.5895,
    0.6320,
    0.5554,
    0.6129,
    0.6774,
    0.5972,
    0.5721,
    0.6739,
    0.6330,
    0.6354,
    0.5866,
    0.5640,
    0.6209,
    0.5695,
    0.5812,
    0.6121,
    0.5797,
    0.5530,
    0.5842,
    0.6131,
    0.6954,
    0.6045,
    0.5786,
    0.5749,
    0.6342,
    0.6435,
    0.6189,
    0.5549,
    0.6794,
    0.5707,
    0.5576,
    0.6289,
    0.5846,
    0.6247,
    0.5422,
    0.6031,
    0.6895,
    0.5680,
    0.5864,
    0.5469,
    0.5703,
    0.6504,
    0.5777,
    0.6413,
    0.6595,
    0.6478,
    0.6278,
    0.6463,
    0.6371,
    0.6525,
    0.6302,
    0.5967,
    0.6620,
    0.6611,
    0.6676,
    0.7040,
    0.6811,
    0.6674,
    0.6530,
    0.5396,
    0.6688,
    0.6692,
    0.6602,
    0.6826,
    0.6548,
    0.6247,
    0.5506,
    0.6220,
    0.6293,
    0.5830,
    0.6597,
    0.5624,
    0.6430,
    0.5506,
    0.6398,
    0.6389,
    0.6621,
    0.5924,
    0.6599,
    0.5956,
    0.6379,
    0.6553,
    0.5515,
    0.5508,
    0.5969,
    0.5997,
    0.5780,
    0.6467,
    0.5691,
    0.6960,
    0.5224,
    0.5372,
    0.5562,
    0.5617,
    0.6155,
    0.5960,
    0.5169,
    0.5892,
    0.5947,
    0.5615,
    0.6593,
    0.5467,
    0.5661,
    0.5652,
    0.6608,
    0.6455,
    0.6577,
    0.6357,
    0.6649,
    0.6495,
    0.6649,
    0.6142,
    0.6142,
    0.6365,
    0.6720,
    0.5775,
    0.6212,
    0.6495,
    0.6328,
    0.5452,
    0.6797,
    0.6505,
    0.6994,
    0.6845,
    0.6206,
    0.6855,
    0.6537,
    0.6760,
    0.6368,
    0.6571,
    0.5510,
    0.6425,
    0.5508,
    0.6496,
    0.5779,
    0.5880,
    0.5956,
    0.6127,
    0.6052,
    0.6227,
    0.6368,
    0.6789,
    0.5568,
    0.6108,
    0.5984,
    0.5672,
    0.5921,
    0.5371,
    0.5808,
    0.5924,
    0.5859,
    0.5684,
    0.5631,
    0.6090,
    0.5915,
    0.6404,
    0.5491,
    0.6693,
    0.6479,
    0.6076,
    0.5757,
    0.6922,
    0.5563,
    0.6487,
    0.5906,
    0.6416,
    0.5532,
    0.6567,
    0.6402,
    0.6267,
    0.5979,
    0.5773,
    0.5692,
    0.5685,
    0.5890,
    0.5340,
    0.5718,
    0.5774,
    0.6267,
    0.6518,
    0.5650,
    0.6073,
    0.6438,
    0.6351,
    0.5882,
    0.6156,
    0.5485,
    0.6558,
    0.6587,
    0.6335,
    0.5678,
    0.6345,
    0.6665,
    0.5571,
    0.6519,
    0.6686,
    0.5569,
    0.5310,
    0.6260,
    0.6748,
    0.6149,
    0.5917,
    0.6177,
    0.6366,
    0.6250,
    0.5606,
    0.6486,
    0.5501,
    0.5457,
    0.5453,
    0.6463,
    0.6856,
    0.5657,
    0.6593,
    0.5645,
    0.6790,
    0.6265,
    0.6641,
    0.6990,
    0.6599,
    0.6605,
    0.6697,
    0.5446,
    0.5847,
    0.5937,
    0.6392,
    0.6420,
    0.6766,
    0.6739,
    0.6432,
    0.5638,
    0.5418,
    0.5577,
    0.5486,
    0.6290,
    0.6171,
    0.5908,
    0.5784,
    0.5865,
    0.5880,
    0.5938,
    0.6941,
    0.5609,
    0.6016,
    0.5707,
    0.5671,
    0.5858,
    0.5720,
    0.6237,
    0.6403,
    0.5416,
    0.6507,
    0.5558,
    0.6685,
    0.6825,
    0.6346,
    0.5408,
    0.6681,
    0.6831,
    0.6152,
    0.6642,
    0.6119,
    0.7095,
    0.6901,
    0.6444,
    0.6540,
    0.5092,
    0.5401,
    0.5475,
    0.6319,
    0.6900,
    0.6539,
    0.5793,
    0.6457,
    0.6327,
    0.7000,
    0.5715,
    0.6506,
    0.6496,
    0.5812,
    0.5878,
    0.6425,
    0.6085,
    0.5706,
    0.6322,
    0.5997,
    0.5778,
    0.5331,
    0.6782,
    0.6185,
    0.6572,
    0.6224,
    0.5783,
    0.6010,
    0.5506,
    0.6788,
    0.6012,
    0.6833,
    0.5861,
    0.6255,
    0.6320,
    0.6572,
    0.5664,
    0.6636,
    0.6311,
    0.6144,
    0.5815,
    0.6166,
    0.6042,
    0.5400,
    0.6258,
    0.5845,
    0.6740,
    0.6406,
    0.6497,
    0.6026,
    0.6099,
    0.5464,
    0.6786,
    0.6546,
    0.6637,
    0.6429,
    0.5942,
    0.5844,
    0.6180,
    0.6100,
    0.6222,
    0.6420,
    0.5793,
    0.5149,
    0.6633,
    0.6250,
    0.6348,
    0.5397,
    0.6557,
    0.5898,
    0.5557,
    0.6745,
    0.5710,
    0.6158,
    0.6494,
    0.5557,
    0.5529,
    0.6747,
    0.6431,
    0.5996,
    0.6305,
    0.5965,
    0.6427,
    0.6013,
    0.6142,
    0.5468,
    0.5358,
    0.6620,
    0.6085,
    0.6178,
    0.5911,
    0.5430,
    0.5851,
    0.6157,
    0.5694,
    0.6024,
    0.5293,
    0.5907,
    0.6249,
    0.5317,
    0.5842,
    0.6225,
    0.5972,
    0.5949,
    0.6515,
    0.6786,
    0.6410,
    0.5697,
    0.6835,
    0.6472,
    0.6421,
    0.6212,
    0.5941,
    0.6665,
    0.6943,
    0.6817,
    0.6487,
    0.6151,
    0.5933,
    0.6647,
    0.6157,
    0.6311,
    0.6534,
    0.5756,
    0.6264,
    0.6283,
    0.5626,
    0.5887,
    0.5697,
    0.6512,
    0.6648,
    0.5748,
    0.6001,
    0.6319,
    0.6503,
    0.6328,
    0.6833,
    0.5465,
    0.6682,
    0.5821,
    0.6903,
    0.5621,
    0.5775,
    0.5409,
    0.6789,
    0.5480,
    0.6377,
    0.6371,
    0.6115,
    0.6509,
    0.6183,
    0.5238,
    0.5832,
    0.5498,
    0.6674,
    0.5410,
    0.6662,
    0.5339,
    0.6861,
    0.6604,
    0.6316,
    0.6555,
    0.6601,
    0.6413,
    0.6904,
    0.6481,
    0.6128,
    0.5465,
    0.6920,
    0.5751,
    0.6839,
    0.5953,
    0.6792,
    0.5612,
    0.6899,
    0.5452,
    0.6413,
    0.5653,
    0.5926,
    0.5646,
    0.5745,
    0.5848,
    0.5665,
    0.5525,
    0.5952,
    0.6371,
    0.6007,
    0.5775,
    0.6346,
    0.5431,
    0.5885,
    0.6420,
    0.6248,
    0.6710,
    0.6600,
    0.6068,
    0.5785,
    0.5994,
    0.5481,
    0.5612,
    0.5529,
    0.6065,
    0.6852,
    0.5572,
    0.5476,
    0.5613,
    0.6373,
    0.5995,
    0.6318,
    0.6750,
    0.5486,
    0.6516,
    0.5738,
    0.5510,
    0.5611,
    0.5871,
    0.5699,
    0.6309,
    0.6587,
    0.6067,
    0.5620,
    0.5600,
    0.6608,
    0.6721,
    0.6399,
    0.5800,
    0.5592,
    0.5531,
    0.6306,
    0.5663,
    0.6420,
    0.4694,
    0.5468,
    0.5935,
    0.6339,
    0.6622,
    0.6213,
    0.6015,
    0.6349,
    0.6302,
    0.6144,
    0.5719,
    0.6285,
    0.5959,
    0.6429,
    0.6064,
    0.5607,
    0.5517,
    0.6277,
    0.6259,
    0.6096,
    0.5843,
    0.6142,
    0.5578,
    0.6325,
    0.6474,
    0.5952,
    0.5771,
    0.5248,
    0.5407,
    0.6540,
    0.5316,
    0.5759,
    0.5515,
    0.6540,
    0.5904,
    0.5816,
    0.5673,
    0.6925,
    0.6283,
    0.5867,
    0.5674,
    0.5736,
    0.5609,
    0.6506,
    0.6358,
    0.5978,
    0.5666,
    0.6333,
    0.6219,
    0.6685,
    0.5768,
    0.7030,
    0.6699,
    0.6168,
    0.5689,
    0.5916,
    0.6245,
    0.5497,
    0.6126,
    0.6312,
    0.5532,
    0.5797,
    0.6494,
    0.6262,
    0.5678,
    0.6601,
    0.5776,
    0.6754,
    0.6892,
    0.5531,
    0.6056,
    0.6381,
    0.5626,
    0.6715,
    0.6402,
    0.6724,
    0.6593,
    0.7040,
    0.5832,
    0.5974,
    0.6673,
    0.6903,
    0.5424,
    0.6411,
    0.6780,
    0.6805,
    0.6771,
    0.6717,
    0.6378,
    0.6346,
    0.5746,
    0.5243,
    0.6565,
    0.5646,
    0.6216,
    0.5819,
    0.5907,
    0.5495,
    0.6175,
    0.5997,
    0.5861,
    0.6880,
    0.7023,
    0.6592,
    0.5509,
    0.5456,
    0.6425,
    0.5994,
]

# 设置直方图的参数
num_bins = 30  # 设置直方图的条数
hist, bins = np.histogram(data, bins=num_bins, density=True)

# 计算每个条的中心位置
bin_centers = 0.5 * (bins[1:] + bins[:-1])

plt.rcParams.update({"font.size": 12})  # 这里可以调整大小，例如12

# 创建3D图形
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

# 绘制柱状图
ax.bar(bin_centers, hist, width=bins[1] - bins[0], zs=0, zdir="y", alpha=0.8)

# 设置坐标轴标签
ax.set_xlabel("值")
ax.set_ylabel("概率密度")
ax.set_zlabel("频率")

# 添加标题
ax.set_title("数据概率分布的三维直方图")

# 显示图形
plt.show()
