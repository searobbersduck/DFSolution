import numpy as np
import math, random, torch, pickle
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


"""Quadratic weighted kappa metric.
   Source: https://github.com/sveitser/kaggle_diabetic/blob/master/quadratic_weighted_kappa.py
   Origin: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py
"""


def kappa_confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
	"""
	Returns the confusion matrix between rater's ratings
	"""
	assert (len(rater_a) == len(rater_b))
	if min_rating is None:
		min_rating = min(rater_a + rater_b)
	if max_rating is None:
		max_rating = max(rater_a + rater_b)
	num_ratings = int(max_rating - min_rating + 1)
	conf_mat = [[0 for i in range(num_ratings)]
	            for j in range(num_ratings)]
	for a, b in zip(rater_a, rater_b):
		conf_mat[a - min_rating][b - min_rating] += 1
	return conf_mat


def kappa_histogram(ratings, min_rating=None, max_rating=None):
	"""
	Returns the counts of each type of rating that a rater made
	"""
	if min_rating is None:
		min_rating = min(ratings)
	if max_rating is None:
		max_rating = max(ratings)
	num_ratings = int(max_rating - min_rating + 1)
	hist_ratings = [0 for x in range(num_ratings)]
	for r in ratings:
		hist_ratings[r - min_rating] += 1
	return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=0, max_rating=4):
	"""
	Calculates the quadratic weighted kappa
	quadratic_weighted_kappa calculates the quadratic weighted kappa
	value, which is a measure of inter-rater agreement between two raters
	that provide discrete numeric ratings.  Potential values range from -1
	(representing complete disagreement) to 1 (representing complete
	agreement).  A kappa value of 0 is expected if all agreement is due to
	chance.

	quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
	each correspond to a list of integer ratings.  These lists must have the
	same lengtorch.

	The ratings should be integers, and it is assumed that they contain
	the complete range of possible ratings.

	quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
	is the minimum possible rating, and max_rating is the maximum possible
	rating
	"""
	rater_a = np.clip(rater_a, min_rating, max_rating)
	rater_b = np.clip(rater_b, min_rating, max_rating)

	rater_a = np.round(rater_a).astype(int).ravel()
	rater_a[~np.isfinite(rater_a)] = 0
	rater_b = np.round(rater_b).astype(int).ravel()
	rater_b[~np.isfinite(rater_b)] = 0

	assert (len(rater_a) == len(rater_b))
	if min_rating is None:
		min_rating = min(min(rater_a), min(rater_b))
	if max_rating is None:
		max_rating = max(max(rater_a), max(rater_b))
	conf_mat = kappa_confusion_matrix(rater_a, rater_b, min_rating, max_rating)
	num_ratings = len(conf_mat)
	num_scored_items = float(len(rater_a))

	hist_rater_a = kappa_histogram(rater_a, min_rating, max_rating)
	hist_rater_b = kappa_histogram(rater_b, min_rating, max_rating)

	numerator = 0.0
	denominator = 0.0

	for i in range(num_ratings):
		for j in range(num_ratings):
			expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
			d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
			numerator += d * conf_mat[i][j] / num_scored_items
			denominator += d * expected_count / num_scored_items
	if denominator<1e-11:
		return -1.0
	else:
		return 1.0 - numerator / denominator


def Tensor2PILImage(pic):
	npimg = pic
	mode = None
	if isinstance(pic, torch.FloatTensor):
		pic = pic.mul(255).byte()
	if torch.is_tensor(pic):
		npimg = np.transpose(pic.numpy(), (1, 2, 0))
	assert isinstance(npimg, np.ndarray), 'pic should be Tensor or ndarray'
	if npimg.shape[2] == 1:
		npimg = npimg[:, :, 0]

		if npimg.dtype == np.uint8:
			mode = 'L'
		if npimg.dtype == np.int16:
			mode = 'I;16'
		if npimg.dtype == np.int32:
			mode = 'I'
		elif npimg.dtype == np.float32:
			mode = 'F'
	else:
		if npimg.dtype == np.uint8:
			mode = 'RGB'
	assert mode is not None, '{} is not supported'.format(npimg.dtype)
	return Image.fromarray(npimg, mode=mode)


def PILImage2Tensor(pic):
	if isinstance(pic, np.ndarray):
		# handle numpy array
		img = torch.from_numpy(pic.transpose((2, 0, 1)))
		# backward compatibility
		return img.float().div(255)

	# handle PIL Image
	if pic.mode == 'I':
		img = torch.from_numpy(np.array(pic, np.int32, copy=False))
	elif pic.mode == 'I;16':
		img = torch.from_numpy(np.array(pic, np.int16, copy=False))
	else:
		img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
	# PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
	if pic.mode == 'YCbCr':
		nchannel = 3
	elif pic.mode == 'I;16':
		nchannel = 1
	else:
		nchannel = len(pic.mode)
	img = img.view(pic.size[1], pic.size[0], nchannel)
	# put it from HWC to CHW format
	# yikes, this transpose takes 80% of the loading time/CPU
	img = img.transpose(0, 1).transpose(0, 2).contiguous()
	if isinstance(img, torch.ByteTensor):
		return img.float().div(255)
	else:
		return img


class TenCrop(object):
	def __init__(self, index, crop_size, scale_size):
		if index == 0:
			self.crop = ((scale_size - crop_size) // 2, (scale_size - crop_size) // 2,
			             (scale_size + crop_size) // 2, (scale_size + crop_size) // 2)
		elif index == 1:
			self.crop = (0, 0, crop_size, crop_size)
		elif index == 2:
			self.crop = (scale_size - crop_size, 0, scale_size, crop_size)
		elif index == 3:
			self.crop = (0, scale_size - crop_size, crop_size, scale_size)
		elif index == 4:
			self.crop = (scale_size - crop_size, scale_size - crop_size, scale_size, scale_size)

	def __call__(self, img):
		return img.crop(self.crop)


class HorizontalFlip(object):
	def __init__(self, flag):
		self.flag = flag

	def __call__(self, img):
		if self.flag:
			return img.transpose(Image.FLIP_LEFT_RIGHT)
		return img


class GaussianBlur(object):
	def __call__(self, image):
		image = image.filter(ImageFilter.GaussianBlur)
		return image


class RandomBlur(object):
	def __init__(self, prob=0.5):
		self.prob = prob
	def __call__(self, image):
		if random.random() < self.prob:
			image = image.filter(ImageFilter.BLUR)
		return image


class RandomBrightness(object):
	def __init__(self, var=0.4):
		self.var = var

	def __call__(self, image):
		alpha = 1.0 + np.random.uniform(-self.var, self.var)
		image = ImageEnhance.Brightness(image).enhance(alpha)
		return image


class RandomColor(object):
	def __init__(self, var=0.4):
		self.var = var

	def __call__(self, image):
		alpha = 1.0 + np.random.uniform(-self.var, self.var)
		image = ImageEnhance.Color(image).enhance(alpha)
		return image


class RandomContrast(object):
	def __init__(self, var=0.4):
		self.var = var

	def __call__(self, image):
		alpha = 1.0 + np.random.uniform(-self.var, self.var)
		image = ImageEnhance.Contrast(image).enhance(alpha)
		return image


class RandomSharpness(object):
	def __init__(self, var=0.4):
		self.var = var

	def __call__(self, image):
		alpha = 1.0 + np.random.uniform(-self.var, self.var)
		image = ImageEnhance.Sharpness(image).enhance(alpha)
		return image


class PILColorJitter(object):
	def __init__(self, blur=0.5, brightness=0.4, color=0.4, contrast=0.4, sharpness=0.4):
		self.transforms = [RandomBlur(blur)]
		if brightness > 0:
			self.transforms.append(RandomBrightness(brightness))
		if color > 0:
			self.transforms.append(RandomColor(color))
		if contrast > 0:
			self.transforms.append(RandomContrast(contrast))
		if sharpness > 0:
			self.transforms.append(RandomSharpness(sharpness))

	def __call__(self, img):
		if self.transforms:
			order = torch.randperm(len(self.transforms))
			for i in order:
				img = self.transforms[i](img)
		return img


# Source: https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py

class Lighting(object):
	"""Lighting noise(AlexNet - style PCA - based noise)"""

	def __init__(self, alphastd, eigval, eigvec):
		self.alphastd = alphastd
		self.eigval = eigval
		self.eigvec = eigvec

	def __call__(self, img):
		if self.alphastd == 0:
			return img
		alpha = img.new().resize_(3).normal_(0, self.alphastd)
		rgb = self.eigvec.type_as(img).clone() \
			.mul(alpha.view(1, 3).expand(3, 3)) \
			.mul(self.eigval.view(1, 3).expand(3, 3)) \
			.sum(1).squeeze()

		return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):
	def __call__(self, img):
		gs = img.clone()
		gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
		gs[1].copy_(gs[0])
		gs[2].copy_(gs[0])
		return gs


class Saturation(object):
	def __init__(self, var):
		self.var = var

	def __call__(self, img):
		gs = Grayscale()(img)
		alpha = random.uniform(0, self.var)
		return img.lerp(gs, alpha)


class Brightness(object):
	def __init__(self, var):
		self.var = var

	def __call__(self, img):
		gs = img.new().resize_as_(img).zero_()
		alpha = random.uniform(0, self.var)
		return img.lerp(gs, alpha)


class Contrast(object):
	def __init__(self, var):
		self.var = var

	def __call__(self, img):
		gs = Grayscale()(img)
		gs.fill_(gs.mean())
		alpha = random.uniform(0, self.var)
		return img.lerp(gs, alpha)


class RandomOrder(object):
	""" Composes several transforms together in random order.
	"""

	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, img):
		if self.transforms is None:
			return img
		order = torch.randperm(len(self.transforms))
		for i in order:
			img = self.transforms[i](img)
		return img


class ColorJitter(object):
	def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
		self.transforms = []
		if brightness > 0:
			self.transforms.append(Brightness(brightness))
		if contrast > 0:
			self.transforms.append(Contrast(contrast))
		if saturation > 0:
			self.transforms.append(Saturation(saturation))

	def __call__(self, img):
		if self.transforms:
			order = torch.randperm(len(self.transforms))
			for i in order:
				img = self.transforms[i](img)
		return img


# Source: https://github.com/ncullen93/torchsample/blob/master/torchsample/utils.py

def th_allclose(x, y):
	"""
	Determine whether two torch tensors have same values
	Mimics np.allclose
	"""
	return torch.sum(torch.abs(x - y)) < 1e-5


def th_flatten(x):
	"""Flatten tensor"""
	return x.contiguous().view(-1)


def th_c_flatten(x):
	"""
	Flatten tensor, leaving channel intact.
	Assumes CHW format.
	"""
	return x.contiguous().view(x.size(0), -1)


def th_bc_flatten(x):
	"""
	Flatten tensor, leaving batch and channel dims intact.
	Assumes BCHW format
	"""
	return x.contiguous().view(x.size(0), x.size(1), -1)


def th_iterproduct(*args):
	return torch.from_numpy(np.indices(args).reshape((len(args), -1)).T)


def th_iterproduct_like(x):
	return th_iterproduct(*x.size())


def th_gather_nd(x, coords):
	inds = coords.mv(torch.LongTensor(x.stride()))
	x_gather = torch.index_select(th_flatten(x), 0, inds)
	return x_gather


def th_affine2d(x, matrix, mode='bilinear', center=True):
	"""
	2D Affine image transform on torch.Tensor

	Arguments
	---------
	x : torch.Tensor of size (C, H, W)
		image tensor to be transformed

	matrix : torch.Tensor of size (3, 3) or (2, 3)
		transformation matrix

	mode : string in {'nearest', 'bilinear'}
		interpolation scheme to use

	center : boolean
		whether to alter the bias of the transform 
		so the transform is applied about the center
		of the image rather than the origin

	Example
	------- 
	>>> import torch
	>>> from torchsample.utils import *
	>>> x = torch.zeros(2,1000,1000)
	>>> x[:,100:1500,100:500] = 10
	>>> matrix = torch.FloatTensor([[1.,0,-50],
	...                             [0,1.,-50]])
	>>> xn = th_affine2d(x, matrix, mode='nearest')
	>>> xb = th_affine2d(x, matrix, mode='bilinear')
	"""

	if matrix.dim() == 2:
		matrix = matrix[:2, :]
		matrix = matrix.unsqueeze(0)
	elif matrix.dim() == 3:
		if matrix.size()[1:] == (3, 3):
			matrix = matrix[:, :2, :]

	A_batch = matrix[:, :, :2]
	if A_batch.size(0) != x.size(0):
		A_batch = A_batch.repeat(x.size(0), 1, 1)
	b_batch = matrix[:, :, 2].unsqueeze(1)

	# make a meshgrid of normal coordinates
	_coords = th_iterproduct(x.size(1), x.size(2))
	coords = _coords.unsqueeze(0).repeat(x.size(0), 1, 1).float()

	if center:
		# shift the coordinates so center is the origin
		coords[:, :, 0] = coords[:, :, 0] - (x.size(1) / 2. + 0.5)
		coords[:, :, 1] = coords[:, :, 1] - (x.size(2) / 2. + 0.5)
	# apply the coordinate transformation
	new_coords = coords.bmm(A_batch.transpose(1, 2)) + b_batch.expand_as(coords)

	if center:
		# shift the coordinates back so origin is origin
		new_coords[:, :, 0] = new_coords[:, :, 0] + (x.size(1) / 2. + 0.5)
		new_coords[:, :, 1] = new_coords[:, :, 1] + (x.size(2) / 2. + 0.5)

	# map new coordinates using bilinear interpolation
	if mode == 'nearest':
		x_transformed = th_nearest_interp2d(x, new_coords)
	elif mode == 'bilinear':
		x_transformed = th_bilinear_interp2d(x, new_coords)

	return x_transformed


def th_nearest_interp2d(input, coords):
	"""
	2d nearest neighbor interpolation torch.Tensor
	"""
	# take clamp of coords so they're in the image bounds
	x = torch.clamp(coords[:, :, 0], 0, input.size(1) - 1).round()
	y = torch.clamp(coords[:, :, 1], 0, input.size(2) - 1).round()

	stride = torch.LongTensor(input.stride())
	x_ix = x.mul(stride[1]).long()
	y_ix = y.mul(stride[2]).long()

	input_flat = input.view(input.size(0), -1).contiguous()

	mapped_vals = input_flat.gather(1, x_ix.add(y_ix))

	return mapped_vals.view_as(input)


def th_bilinear_interp2d(input, coords):
	"""
	bilinear interpolation in 2d
	"""
	x = torch.clamp(coords[:, :, 0], 0, input.size(1) - 2)
	x0 = x.floor()
	x1 = x0 + 1
	y = torch.clamp(coords[:, :, 1], 0, input.size(2) - 2)
	y0 = y.floor()
	y1 = y0 + 1

	stride = torch.LongTensor(input.stride())
	x0_ix = x0.mul(stride[1]).long()
	x1_ix = x1.mul(stride[1]).long()
	y0_ix = y0.mul(stride[2]).long()
	y1_ix = y1.mul(stride[2]).long()

	input_flat = input.view(input.size(0), -1).contiguous()

	vals_00 = input_flat.gather(1, x0_ix.add(y0_ix))
	vals_10 = input_flat.gather(1, x1_ix.add(y0_ix))
	vals_01 = input_flat.gather(1, x0_ix.add(y1_ix))
	vals_11 = input_flat.gather(1, x1_ix.add(y1_ix))

	xd = x - x0
	yd = y - y0
	xm = 1 - xd
	ym = 1 - yd

	x_mapped = (vals_00.mul(xm).mul(ym) +
	            vals_10.mul(xd).mul(ym) +
	            vals_01.mul(xm).mul(yd) +
	            vals_11.mul(xd).mul(yd))

	return x_mapped.view_as(input)


def th_affine3d(x, matrix, mode='trilinear', center=True):
	"""
	3D Affine image transform on torch.Tensor
	"""
	A = matrix[:3, :3]
	b = matrix[:3, 3]

	# make a meshgrid of normal coordinates
	coords = th_iterproduct(x.size(1), x.size(2), x.size(3)).float()

	if center:
		# shift the coordinates so center is the origin
		coords[:, 0] = coords[:, 0] - (x.size(1) / 2. + 0.5)
		coords[:, 1] = coords[:, 1] - (x.size(2) / 2. + 0.5)
		coords[:, 2] = coords[:, 2] - (x.size(3) / 2. + 0.5)

	# apply the coordinate transformation
	new_coords = coords.mm(A.t().contiguous()) + b.expand_as(coords)

	if center:
		# shift the coordinates back so origin is origin
		new_coords[:, 0] = new_coords[:, 0] + (x.size(1) / 2. + 0.5)
		new_coords[:, 1] = new_coords[:, 1] + (x.size(2) / 2. + 0.5)
		new_coords[:, 2] = new_coords[:, 2] + (x.size(3) / 2. + 0.5)

	# map new coordinates using bilinear interpolation
	if mode == 'nearest':
		x_transformed = th_nearest_interp3d(x, new_coords)
	elif mode == 'trilinear':
		x_transformed = th_trilinear_interp3d(x, new_coords)
	else:
		x_transformed = th_trilinear_interp3d(x, new_coords)

	return x_transformed


def th_nearest_interp3d(input, coords):
	"""
	2d nearest neighbor interpolation torch.Tensor
	"""
	# take clamp of coords so they're in the image bounds
	coords[:, 0] = torch.clamp(coords[:, 0], 0, input.size(1) - 1).round()
	coords[:, 1] = torch.clamp(coords[:, 1], 0, input.size(2) - 1).round()
	coords[:, 2] = torch.clamp(coords[:, 2], 0, input.size(3) - 1).round()

	stride = torch.LongTensor(input.stride())[1:].float()
	idx = coords.mv(stride).long()

	input_flat = th_flatten(input)

	mapped_vals = input_flat[idx]

	return mapped_vals.view_as(input)


def th_trilinear_interp3d(input, coords):
	"""
	trilinear interpolation of 3D torch.Tensor image
	"""
	# take clamp then floor/ceil of x coords
	x = torch.clamp(coords[:, 0], 0, input.size(1) - 2)
	x0 = x.floor()
	x1 = x0 + 1
	# take clamp then floor/ceil of y coords
	y = torch.clamp(coords[:, 1], 0, input.size(2) - 2)
	y0 = y.floor()
	y1 = y0 + 1
	# take clamp then floor/ceil of z coords
	z = torch.clamp(coords[:, 2], 0, input.size(3) - 2)
	z0 = z.floor()
	z1 = z0 + 1

	stride = torch.LongTensor(input.stride())[1:]
	x0_ix = x0.mul(stride[0]).long()
	x1_ix = x1.mul(stride[0]).long()
	y0_ix = y0.mul(stride[1]).long()
	y1_ix = y1.mul(stride[1]).long()
	z0_ix = z0.mul(stride[2]).long()
	z1_ix = z1.mul(stride[2]).long()

	input_flat = th_flatten(input)

	vals_000 = input_flat[x0_ix + y0_ix + z0_ix]
	vals_100 = input_flat[x1_ix + y0_ix + z0_ix]
	vals_010 = input_flat[x0_ix + y1_ix + z0_ix]
	vals_001 = input_flat[x0_ix + y0_ix + z1_ix]
	vals_101 = input_flat[x1_ix + y0_ix + z1_ix]
	vals_011 = input_flat[x0_ix + y1_ix + z1_ix]
	vals_110 = input_flat[x1_ix + y1_ix + z0_ix]
	vals_111 = input_flat[x1_ix + y1_ix + z1_ix]

	xd = x - x0
	yd = y - y0
	zd = z - z0
	xm1 = 1 - xd
	ym1 = 1 - yd
	zm1 = 1 - zd

	x_mapped = (vals_000.mul(xm1).mul(ym1).mul(zm1) +
	            vals_100.mul(xd).mul(ym1).mul(zm1) +
	            vals_010.mul(xm1).mul(yd).mul(zm1) +
	            vals_001.mul(xm1).mul(ym1).mul(zd) +
	            vals_101.mul(xd).mul(ym1).mul(zd) +
	            vals_011.mul(xm1).mul(yd).mul(zd) +
	            vals_110.mul(xd).mul(yd).mul(zm1) +
	            vals_111.mul(xd).mul(yd).mul(zd))

	return x_mapped.view_as(input)


def th_pearsonr(x, y):
	"""
	mimics scipy.stats.pearsonr
	"""
	mean_x = torch.mean(x)
	mean_y = torch.mean(y)
	xm = x.sub(mean_x)
	ym = y.sub(mean_y)
	r_num = xm.dot(ym)
	r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
	r_val = r_num / r_den
	return r_val


def th_corrcoef(x):
	"""
	mimics np.corrcoef
	"""
	# calculate covariance matrix of rows
	mean_x = torch.mean(x, 1)
	xm = x.sub(mean_x.expand_as(x))
	c = xm.mm(xm.t())
	c = c / (x.size(1) - 1)

	# normalize covariance matrix
	d = torch.diag(c)
	stddev = torch.pow(d, 0.5)
	c = c.div(stddev.expand_as(c))
	c = c.div(stddev.expand_as(c).t())

	# clamp between -1 and 1
	c = torch.clamp(c, -1.0, 1.0)

	return c


def th_matrixcorr(x, y):
	"""
	return a correlation matrix between
	columns of x and columns of y.

	So, if X.size() == (1000,4) and Y.size() == (1000,5),
	then the result will be of size (4,5) with the
	(i,j) value equal to the pearsonr correlation coeff
	between column i in X and column j in Y
	"""
	mean_x = torch.mean(x, 0)
	mean_y = torch.mean(y, 0)
	xm = x.sub(mean_x.expand_as(x))
	ym = y.sub(mean_y.expand_as(y))
	r_num = xm.t().mm(ym)
	r_den1 = torch.norm(xm, 2, 0)
	r_den2 = torch.norm(ym, 2, 0)
	r_den = r_den1.t().mm(r_den2)
	r_mat = r_num.div(r_den)
	return r_mat


def th_random_choice(a, size=None, replace=True, p=None):
	"""
	Parameters
	-----------
	a : 1-D array-like
		If a torch.Tensor, a random sample is generated from its elements.
		If an int, the random sample is generated as if a was torch.range(n)
	size : int, optional
		Number of samples to draw. Default is None, in which case a
		single value is returned.
	replace : boolean, optional
		Whether the sample is with or without replacement
	p : 1-D array-like, optional
		The probabilities associated with each entry in a.
		If not given the sample assumes a uniform distribution over all
		entries in a.

	Returns
	--------
	samples : 1-D ndarray, shape (size,)
		The generated random samples
	"""
	if size is None:
		size = 1

	if isinstance(a, int):
		a = torch.arange(0, a)

	if p is None:
		if replace:
			idx = torch.floor(torch.rand(size) * a.size(0)).long()
		else:
			idx = torch.randperm(a.size(0))[:size]
	else:
		if abs(1.0 - sum(p)) > 1e-3:
			raise ValueError('p must sum to 1.0')
		if not replace:
			raise ValueError('replace must equal true if probabilities given')
		idx_vec = torch.cat([torch.zeros(round(p[i] * 1000)) + i for i in range(len(p))])
		idx = (torch.floor(torch.rand(size) * 999.99)).long()
		idx = idx_vec[idx].long()
	return a[idx]


def save_transform(file, transform):
	"""
	Save a transform object
	"""
	with open(file, 'wb') as output_file:
		pickler = pickle.Pickler(output_file, -1)
		pickler.dump(transform)


def load_transform(file):
	"""
	Load a transform object
	"""
	with open(file, 'rb') as input_file:
		transform = pickle.load(input_file)
	return transform


# Source: https://github.com/ncullen93/torchsample/blob/master/torchsample/transforms/affine_transforms.py

class Affine(object):
	def __init__(self,
	             rotation_range=None,
	             translation_range=None,
	             shear_range=None,
	             zoom_range=None):
		"""
		Perform an affine transforms with various sub-transforms, using
		only one interpolation and without having to instantiate each
		sub-transform individually.

		Arguments
		---------
		rotation_range : one integer or float
			image will be rotated between (-degrees, degrees) degrees

		translation_range : a float or a tuple/list w/ 2 floats between [0, 1)
			first value:
				image will be horizontally shifted between 
				(-height_range * height_dimension, height_range * height_dimension)
			second value:
				Image will be vertically shifted between 
				(-width_range * width_dimension, width_range * width_dimension)

		shear_range : float
			radian bounds on the shear transform

		zoom_range : list/tuple with two floats between [0, infinity).
			first float should be less than the second
			lower and upper bounds on percent zoom. 
			Anything less than 1.0 will zoom in on the image, 
			anything greater than 1.0 will zoom out on the image.
			e.g. (0.7, 1.0) will only zoom in, 
				 (1.0, 1.4) will only zoom out,
				 (0.7, 1.4) will randomly zoom in or out

		fill_mode : string in {'constant', 'nearest'}
			how to fill the empty space caused by the transform
			ProTip : use 'nearest' for discrete images (e.g. segmentations)
					and use 'constant' for continuous images

		fill_value : float
			the value to fill the empty space with if fill_mode='constant'

		target_fill_mode : same as fill_mode, but for target image

		target_fill_value : same as fill_value, but for target image

		"""
		self.transforms = []
		if rotation_range is not None:
			rotation_tform = Rotate(rotation_range, lazy=True)
			self.transforms.append(rotation_tform)

		if translation_range is not None:
			translation_tform = Translate(translation_range, lazy=True)
			self.transforms.append(translation_tform)

		if shear_range is not None:
			shear_tform = Shear(shear_range, lazy=True)
			self.transforms.append(shear_tform)

		if zoom_range is not None:
			zoom_tform = Zoom(zoom_range, lazy=True)
			self.transforms.append(zoom_tform)

		if len(self.transforms) == 0:
			raise Exception('Must give at least one transform parameter in Affine()')

	def __call__(self, x, y=None):
		# collect all of the lazily returned tform matrices
		tform_matrix = self.transforms[0](x)
		for tform in self.transforms[1:]:
			tform_matrix = torch.mm(tform_matrix, tform(x))

		x = th_affine2d(x, tform_matrix)

		self.tform_matrix = tform_matrix

		if y is not None:
			y = th_affine2d(y, tform_matrix)
			return x, y
		else:
			return x


class AffineCompose(object):
	def __init__(self,
	             transforms,
	             fixed_size=None):
		"""
		Apply a collection of explicit affine transforms to an input image,
		and to a target image if necessary

		Arguments
		---------
		transforms : list or tuple
			each element in the list/tuple should be an affine transform.
			currently supported transforms:
				- Rotate()
				- Translate()
				- Shear()
				- Zoom()

		fill_mode : string in {'constant', 'nearest'}
			how to fill the empty space caused by the transform

		fill_value : float
			the value to fill the empty space with if fill_mode='constant'

		"""
		self.transforms = transforms
		# set transforms to lazy so they only return the tform matrix
		for t in self.transforms:
			t.lazy = True

		# self.coords = None
		# if fixed_size is not None:
		#    if len(fixed_size) == 3:
		#        # assume channel is first dim
		#        fixed_size = fixed_size[1:]
		#    self.coords = th_iterproduct(fixed_size[0], fixed_size[1])

	def __call__(self, x, y=None):
		# collect all of the lazily returned tform matrices
		tform_matrix = self.transforms[0](x)
		for tform in self.transforms[1:]:
			tform_matrix = torch.mm(tform_matrix, tform(x))

		x = th_affine2d(x, tform_matrix)  # , self.coords)

		if y is not None:
			y = th_affine2d(y, tform_matrix)  # , self.coords)
			return x, y
		else:
			return x


class Rotate(object):
	def __init__(self,
	             rotation_range,
	             fixed_size=None,
	             lazy=False):
		"""
		Randomly rotate an image between (-degrees, degrees). If the image
		has multiple channels, the same rotation will be applied to each channel.

		Arguments
		---------
		rotation_range : integer or float
			image will be rotated between (-degrees, degrees) degrees

		fill_mode : string in {'constant', 'nearest'}
			how to fill the empty space caused by the transform

		fill_value : float
			the value to fill the empty space with if fill_mode='constant'

		lazy    : boolean
			if false, perform the transform on the tensor and return the tensor
			if true, only create the affine transform matrix and return that
		"""
		self.rotation_range = rotation_range
		self.lazy = lazy

	# self.coords = None
	# if not self.lazy and fixed_size is not None:
	#    self.coords = th_iterproduct(fixed_size[0], fixed_size[1])

	def __call__(self, x, y=None):
		degree = random.uniform(-self.rotation_range, self.rotation_range)
		theta = math.pi / 180 * degree
		rotation_matrix = torch.FloatTensor([[math.cos(theta), -math.sin(theta), 0],
		                                     [math.sin(theta), math.cos(theta), 0],
		                                     [0, 0, 1]])
		if self.lazy:
			return rotation_matrix
		else:
			x_transformed = th_affine2d(x, rotation_matrix)  # , self.coords)
			if y is not None:
				y_transformed = th_affine2d(y, rotation_matrix)  # , self.coords)
				return x_transformed, y_transformed
			else:
				return x_transformed


class Translate(object):
	def __init__(self,
	             translation_range,
	             fixed_size=None,
	             lazy=False):
		"""
		Randomly translate an image some fraction of total height and/or
		some fraction of total width. If the image has multiple channels,
		the same translation will be applied to each channel.

		Arguments
		---------
		translation_range : two floats between [0, 1) 
			first value:
				fractional bounds of total height to shift image
				image will be horizontally shifted between 
				(-height_range * height_dimension, height_range * height_dimension)
			second value:
				fractional bounds of total width to shift image 
				Image will be vertically shifted between 
				(-width_range * width_dimension, width_range * width_dimension)

		fill_mode : string in {'constant', 'nearest'}
			how to fill the empty space caused by the transform

		fill_value : float
			the value to fill the empty space with if fill_mode='constant'

		lazy    : boolean
			if false, perform the transform on the tensor and return the tensor
			if true, only create the affine transform matrix and return that
		"""
		if isinstance(translation_range, float):
			translation_range = (translation_range, translation_range)
		self.height_range = translation_range[0]
		self.width_range = translation_range[1]
		self.lazy = lazy

	# self.coords = None
	# if not self.lazy and fixed_size is not None:
	#    self.coords = th_iterproduct(fixed_size[0], fixed_size[1])

	def __call__(self, x, y=None):
		# height shift
		if self.height_range > 0:
			tx = random.uniform(-self.height_range, self.height_range) * x.size(1)
		else:
			tx = 0
		# width shift
		if self.width_range > 0:
			ty = random.uniform(-self.width_range, self.width_range) * x.size(2)
		else:
			ty = 0

		translation_matrix = torch.FloatTensor([[1, 0, tx],
		                                        [0, 1, ty],
		                                        [0, 0, 1]])
		if self.lazy:
			return translation_matrix
		else:
			x_transformed = th_affine2d(x, translation_matrix)  # , self.coords)
			if y is not None:
				y_transformed = th_affine2d(y, translation_matrix)  # , self.coords)
				return x_transformed, y_transformed
			else:
				return x_transformed


class Shear(object):
	def __init__(self,
	             shear_range,
	             fixed_size=None,
	             lazy=False):
		"""
		Randomly shear an image with radians (-shear_range, shear_range)

		Arguments
		---------
		shear_range : float
			radian bounds on the shear transform

		fill_mode : string in {'constant', 'nearest'}
			how to fill the empty space caused by the transform

		fill_value : float
			the value to fill the empty space with if fill_mode='constant'

		lazy    : boolean
			if false, perform the transform on the tensor and return the tensor
			if true, only create the affine transform matrix and return that
		"""
		self.shear_range = shear_range
		self.lazy = lazy

	# self.coords = None
	# if not self.lazy and fixed_size is not None:
	#    self.coords = th_iterproduct(fixed_size[0], fixed_size[1])

	def __call__(self, x, y=None):
		shear = random.uniform(-self.shear_range, self.shear_range)
		shear_matrix = torch.FloatTensor([[1, -math.sin(shear), 0],
		                                  [0, math.cos(shear), 0],
		                                  [0, 0, 1]])
		if self.lazy:
			return shear_matrix
		else:
			x_transformed = th_affine2d(x, shear_matrix)  # , self.coords)
			if y is not None:
				y_transformed = th_affine2d(y, shear_matrix)  # , self.coords)
				return x_transformed, y_transformed
			else:
				return x_transformed


class Zoom(object):
	def __init__(self,
	             zoom_range,
	             fixed_size=None,
	             lazy=False):
		"""
		Randomly zoom in and/or out on an image 

		Arguments
		---------
		zoom_range : tuple or list with 2 values, both between (0, infinity)
			lower and upper bounds on percent zoom. 
			Anything less than 1.0 will zoom in on the image, 
			anything greater than 1.0 will zoom out on the image.
			e.g. (0.7, 1.0) will only zoom in, 
				 (1.0, 1.4) will only zoom out,
				 (0.7, 1.4) will randomly zoom in or out

		fill_mode : string in {'constant', 'nearest'}
			how to fill the empty space caused by the transform

		fill_value : float
			the value to fill the empty space with if fill_mode='constant'

		lazy    : boolean
			if false, perform the transform on the tensor and return the tensor
			if true, only create the affine transform matrix and return that
		"""
		if not isinstance(zoom_range, list) and not isinstance(zoom_range, tuple):
			raise ValueError('zoom_range must be tuple or list with 2 values')
		self.zoom_range = zoom_range
		self.lazy = lazy

	# self.coords = None
	# if not self.lazy and fixed_size is not None:
	#    self.coords = th_iterproduct(fixed_size[0], fixed_size[1])


	def __call__(self, x, y=None):
		zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
		zy = random.uniform(self.zoom_range[0], self.zoom_range[1])
		zoom_matrix = torch.FloatTensor([[zx, 0, 0],
		                                 [0, zy, 0],
		                                 [0, 0, 1]])
		if self.lazy:
			return zoom_matrix
		else:
			x_transformed = th_affine2d(x, zoom_matrix)  # , self.coords)
			if y is not None:
				y_transformed = th_affine2d(y, zoom_matrix)  # , self.coords)
				return x_transformed, y_transformed
			else:
				return x_transformed
