import numpy as np
import torch
import torch.nn as nn
import math
# GPU-accelerated KMeans using cuML (RAPIDS)
try:
    from cuml import KMeans as cuMLKMeans
    CUML_AVAILABLE = True
    print("Using cuML KMeans for GPU acceleration")
except ImportError:
    CUML_AVAILABLE = False
    print("Warning: cuML not available, falling back to CPU sklearn. Install cuML for GPU acceleration: pip install cuml")

import torch
from torch.distributions import Normal

def gpu_kmeans_clustering(data, n_clusters, sample_weight=None, device=None, max_iter=50):
    """
    GPU-accelerated KMeans clustering using cuML (RAPIDS)
    """
    if device is None:
        device = data.device
    
    if CUML_AVAILABLE:
        try:
            # Convert PyTorch tensor to CuPy array for cuML (more efficient GPU->GPU transfer)
            import cupy as cp
            
            if data.is_cuda:
                # Direct GPU memory access without CPU transfer
                data_cupy = cp.asarray(data.detach())
            else:
                # If data is on CPU, convert via numpy
                data_cupy = cp.asarray(data.detach().numpy())
            
            # Create cuML KMeans with sklearn-compatible API
            kmeans_cuml = cuMLKMeans(
                n_clusters=n_clusters, 
                max_iter=max_iter, 
                random_state=0,
                n_init='auto'
            )
            
            if sample_weight is not None:
                if sample_weight.is_cuda:
                    sample_weight_cupy = cp.asarray(sample_weight.detach())
                else:
                    sample_weight_cupy = cp.asarray(sample_weight.detach().numpy())
                kmeans_cuml.fit(data_cupy, sample_weight=sample_weight_cupy)
            else:
                kmeans_cuml.fit(data_cupy)
            
            # Convert cluster centers back to numpy
            return cp.asnumpy(kmeans_cuml.cluster_centers_)
                
        except ImportError as e:
            print(f"cuML or CuPy import failed, falling back to CPU sklearn: {e}")
        except Exception as e:
            print(f"cuML KMeans failed, falling back to CPU sklearn: {e}")
    
    # Fallback to sklearn CPU implementation
    from sklearn.cluster import KMeans
    data_cpu = data.cpu().numpy()
    if sample_weight is not None:
        sample_weight_cpu = sample_weight.cpu().numpy()
        kmeans_sklearn = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto", max_iter=max_iter)
        kmeans_sklearn.fit(data_cpu, sample_weight=sample_weight_cpu)
    else:
        kmeans_sklearn = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto", max_iter=max_iter)
        kmeans_sklearn.fit(data_cpu)
    
    return kmeans_sklearn.cluster_centers_

def round_to_nearest_pole_sim(w, poles, return_freq=False):
    """
    w: weight/act values (1d vector)
    poles: tuple of values
    return_freq: whether to also return frequency of each pole usage

    Round the numbers in w to the nearest value in poles.
    """
    stack = []
    for c in poles:
        diff = (w - c).abs()
        stack.append(diff)
    diff = torch.stack(stack)
    idx = diff.argmin(axis=0)
    aug = 0
    freq = []
    for i, c in enumerate(poles):
        mask = (idx == i)
        aug += mask * c
        if return_freq:
            freq.append(mask.sum().item())

    if return_freq:
        return aug, freq
    else:
        return aug

def get_outliers(
    w,
    channel=-1,
    outlier_threshold_upper=-1,
    outlier_threshold_lower=-1,
    cap_outliers=-1,
    first_few_fp16=-1
):
    """
    w: weight/act values (1d vector)
    channel: which dimension to share scaling factors along
    outlier_threshold_upper: upper outlier thresholds
    outlier_threshold_lower: lower outlier thresholds
    first_few_fp16: number of initial tokens to keep in fp16

    Detect outliers above upper threshold / below lower threshold
    """
    # only use either per-channel or per-token outlier
    outlier_threshold_upper = outlier_threshold_upper.unsqueeze(channel)
    outlier_threshold_lower = outlier_threshold_lower.unsqueeze(channel)

    under_lower = w < outlier_threshold_lower
    above_upper = w > outlier_threshold_upper

    outlier_mask = torch.logical_or(under_lower, above_upper)

    if cap_outliers > -1:
        outlier_mask_tmp = outlier_mask.clone()

        zero_point = (outlier_threshold_upper + outlier_threshold_lower) / 2
        distance = (outlier_threshold_upper - outlier_threshold_lower) / 2
        outliers = w * outlier_mask

        values = torch.zeros_like(outliers)
        values[outlier_mask] = ((w - zero_point) / distance)[outlier_mask]

        upper_values, upper_indices = torch.topk(values, 21, dim=-1)
        lower_values, lower_indices = torch.topk(values, 21, dim=-1, largest=False)
        indices_combined = torch.cat((upper_indices, lower_indices), dim=-1)
        values_combined = torch.cat((upper_values, lower_values), dim=-1)

        values2 = torch.zeros_like(outliers)
        values2.scatter_(-1, indices_combined, values_combined)
        outlier_mask = values2 != 0

    if first_few_fp16 > -1:
        outlier_mask[:first_few_fp16,:] = True

    return outlier_mask

def get_outliers_dynamic(
    w,
    channel=-1,
    thresh=0.999,
    first_few_fp16=-1
):
    """
    w: weight/act values (1d vector)
    channel: which dimension to share scaling factors along
    thresh: percentile for outlier threshold computation
    first_few_fp16: number of initial tokens to keep in fp16

    Detect outliers above upper threshold / below lower threshold
    """

    t = 1-((1-thresh)/2)
    w = w.float()

    # only use either per-channel or per-token outlier
    outlier_threshold_upper = torch.quantile(w, t, dim=channel)
    outlier_threshold_lower = torch.quantile(w, 1-t, dim=channel)

    outlier_threshold_upper = outlier_threshold_upper.unsqueeze(channel)
    outlier_threshold_lower = outlier_threshold_lower.unsqueeze(channel)

    under_lower = w <= outlier_threshold_lower
    above_upper = w >= outlier_threshold_upper

    outlier_mask = torch.logical_or(under_lower, above_upper)

    if first_few_fp16 > -1:
        outlier_mask[:first_few_fp16,:] = True

    return outlier_mask

# integer quantization function
def quant_fn_zp(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    clamp=False,
    group_size=-1,
    one_group=False
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)
    clamp: whether to round and clamp the zeropoint

    Performs simulated integer quantization
    """

    # set quantization threshold dynamically
    if dynamicquantization or group_size > 0:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using median to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            if group_size > 0 and not one_group:
                inp_shape = inp.shape
                num_groups = inp_shape[qchannel] // group_size
                if qchannel == 0:
                    inp = inp.reshape(num_groups, group_size, -1)
                    maxval = torch.max(inp, dim=1, keepdim=True).values
                    minval = torch.min(inp, dim=1, keepdim=True).values
                else:
                    inp = inp.reshape(-1, num_groups, group_size)
                    maxval = torch.max(inp, dim=-1, keepdim=True).values
                    minval = torch.min(inp, dim=-1, keepdim=True).values
            else:
                maxval = torch.max(inp, dim=qchannel, keepdim=True).values
                minval = torch.min(inp, dim=qchannel, keepdim=True).values

    # compute offset here:
    rangeval = (maxval - minval)
    qx = (2**bits - 1) / rangeval

    # set offset
    if clamp:
        offset = torch.round(minval * qx)
        offset = offset.clamp(-(2**bits - 1), 0)
    else: # improves accuracy with per-channel key quantization
        offset = minval * qx

    # need to handle outlier removal
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    # scale and subtract offset
    qinp = torch.round(qx * inp - offset)

    #clipping (just for debugging purposes)
    qinp = torch.clip(qinp, min=0, max=2**bits - 1)

    #rescale
    qinp_out = (qinp + offset) / qx

    # add outliers back
    if include_sparse:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers

    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)
    return qinp_out

def quant_fn_nf(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    nf_lut=None,
    group_size=-1,
    one_group=False
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)
    nf_lut: NormalFloat signpost values

    Performs simulated NormalFloat quantization
    """

    # set quantization threshold dynamically
    if dynamicquantization or group_size > 0:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using mean to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            if group_size > 0 and not one_group:
                inp_shape = inp.shape
                num_groups = inp_shape[qchannel] // group_size
                if qchannel == 0:
                    inp = inp.reshape(num_groups, group_size, -1)
                    maxval = torch.max(inp, dim=1, keepdim=True).values
                    minval = torch.min(inp, dim=1, keepdim=True).values
                else:
                    inp = inp.reshape(-1, num_groups, group_size)
                    maxval = torch.max(inp, dim=-1, keepdim=True).values
                    minval = torch.min(inp, dim=-1, keepdim=True).values
            else:
                maxval = torch.max(inp, dim=qchannel, keepdim=True).values
                minval = torch.min(inp, dim=qchannel, keepdim=True).values

    # compute offset here:
    offset = (maxval + minval) / 2
    rangeval = (maxval - minval) / 2

    # subtract offset
    inp = inp - offset

    # need to handle outlier removal here due to issues with zeroing out non-outliers
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    #dividing by range to normalize to [-1,1]
    inp_scaled = inp / rangeval

    Q = round_to_nearest_pole_sim(inp_scaled.flatten(), nf_lut)
    qinp_out = Q.reshape(inp.shape).half().cuda()
    qinp_out = qinp_out * rangeval

    # add outliers back
    if include_sparse:
        qinp_out = qinp_out + outliers

    #shift by offset
    qinp_out = qinp_out + offset
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0) #TODO: debug (shouldn't be necessary)

    return qinp_out

def quant_fn_nuq_recon(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    lut=None,
    norm=False,
    normscale=None,
    normoffset=None,
    first_few_fp16=-1,
    group_size=-1,
    one_group=False
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)
    lut: NUQ signpost values
    norm: whether to use Q-Norm
    normscale: scaling for Q-Norm
    normoffset: shift for Q-Norm
    first_few_fp16: number of initial tokens to keep in fp16

    Performs simulated NUQ quantization
    """

    if first_few_fp16 > -1:
        orig = inp

    # set quantization threshold dynamically
    if dynamicquantization or group_size > 0:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using mean to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            if group_size > 0 and not one_group:
                inp_shape = inp.shape
                num_groups = inp_shape[qchannel] // group_size
                if qchannel == 0:
                    inp = inp.reshape(num_groups, group_size, -1)
                    maxval = torch.max(inp, dim=1, keepdim=True).values
                    minval = torch.min(inp, dim=1, keepdim=True).values
                else:
                    inp = inp.reshape(-1, num_groups, group_size)
                    maxval = torch.max(inp, dim=-1, keepdim=True).values
                    minval = torch.min(inp, dim=-1, keepdim=True).values
            else:
                maxval = torch.max(inp, dim=qchannel, keepdim=True).values
                minval = torch.min(inp, dim=qchannel, keepdim=True).values

    # compute offset here:
    offset = (maxval + minval) / 2
    rangeval = (maxval - minval) / 2

    # subtract offset
    inp = inp - offset

    # need to handle outlier removal here due to issues with zeroing out non-outliers
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    #dividing by range to normalize to [-1,1]
    inp_scaled = inp / rangeval

    # round to nearest LUT entry
    lut_cuda = torch.tensor(lut[0]).to(inp_scaled.device)
    Q = round_to_nearest_pole_sim(inp_scaled.flatten(), lut_cuda)
    qinp_out = Q.reshape(inp.shape).float().to(inp_scaled.device)

    if norm and (group_size == 0 or one_group):
        normscale = normscale.to(inp_scaled.device)
        normoffset = normoffset.to(inp_scaled.device)
        qinp_out = qinp_out*normscale + normoffset

    # un-normalize
    qinp_out = qinp_out * rangeval

    # add outliers back
    if include_sparse:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers

    #shift by offset
    qinp_out = qinp_out + offset
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0) #TODO: debug (shouldn't be necessary)

    # leave first few in fp16
    # leave this here for now -> avoids any small perturbations from rescaling
    if first_few_fp16 > -1:
        qinp_out[:first_few_fp16,:] = orig[:first_few_fp16,:]

    return qinp_out.float()

# simquant quantizer (calibration)
class SimQuant:
    def __init__(
                    self,
                    layer,
                    bits,
                    perchannel=True,
                    qchannel=0,
                    include_rope=False,
                    seqlen=2048,
                    group_size=-1,
                    one_group=False
                ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.perchannel = perchannel
        self.qchannel = qchannel
        self.bits = bits
        self.seqlen = seqlen  # Store sequence length
        
        # Group-wise quantization parameters
        self.group_size = group_size
        self.one_group = one_group

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.nsamples = 0

        self.out = None

    def add_batch(self, inp, out):
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        tmp = out.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(out.shape) == 3:
                out = out.reshape((-1, self.rows))
        self.nsamples += tmp

        if self.out == None:
            self.out = out.clone()
        else:
            self.out = torch.cat((self.out, out.clone()), dim=0)

    def quantize(
        self,
        include_sparse=False,
        sparsity_threshold=0.999,
        nuq=False,
        fisher=False,
        norm=False,
        cap_outliers=False,
        first_few_fp16=-1
    ):

        # for now, just update threshold here
        if include_sparse:
            t = 1-((1-sparsity_threshold)/2)
        else:
            t = 1 #use min-max quantization

        #TODO - if not using sparsity, use a different threshold for min-max quant?
        # Keep data on GPU throughout
        data = self.out.float()
        device = data.device

        if self.perchannel and cap_outliers:
            #per-channel - remove tokenwise outliers and normalize range to [-1,1]
            
            # Use torch.quantile instead of np.percentile for GPU acceleration
            outlier_threshold_upper = torch.quantile(data, t, dim=self.qchannel).unsqueeze(self.qchannel)
            outlier_threshold_lower = torch.quantile(data, 1-t, dim=self.qchannel).unsqueeze(self.qchannel)
            zero_point = (outlier_threshold_upper + outlier_threshold_lower) / 2
            distance = (outlier_threshold_upper - outlier_threshold_lower) / 2
            data2 = ((data - zero_point) / distance).abs()

            outlier_mask = torch.zeros_like(data2, dtype=torch.bool)
            hidden_dim = data.shape[-1]
            num_elems = math.ceil((1-t) * hidden_dim)
            upper_indices = torch.topk(data2, num_elems).indices
            lower_indices = torch.topk(data2, num_elems, largest=False).indices

            true_mask = torch.ones_like(upper_indices, dtype=torch.bool)
            outlier_mask.scatter_(-1, lower_indices, true_mask)
            outlier_mask.scatter_(-1, upper_indices, true_mask)

            if first_few_fp16 > -1 :
                # remove first few tokens
                for i in range(0,self.nsamples):
                    start = i*self.seqlen
                    end = i*self.seqlen + first_few_fp16
                    outlier_mask[start:end,:] = True

            med = torch.median(data, dim=0).values.unsqueeze(0).repeat(data.shape[0],1)
            data_trimmed = data.clone()
            data_trimmed[outlier_mask] = med[outlier_mask] 

            outlier_threshold_upper = torch.max(data_trimmed, axis=self.qchannel).values
            outlier_threshold_lower = torch.min(data_trimmed, axis=self.qchannel).values

            # recomputing outlier mask here before doing k-means fitting
            zero_point = (outlier_threshold_upper + outlier_threshold_lower) / 2
            distance = (outlier_threshold_upper - outlier_threshold_lower) / 2
            zero_point = zero_point.unsqueeze(0)
            distance = distance.unsqueeze(0)
            data_shifted_normalized = ((data - zero_point) / distance).abs()
            outlier_mask = torch.logical_or((data_shifted_normalized > 1), (data_shifted_normalized < -1))

        if self.perchannel:
            #per-channel - remove tokenwise outliers and normalize range to [-1,1]
            # Use torch.quantile instead of np.percentile for GPU acceleration
            if self.group_size > 0 and not self.one_group:
                data_shape = data.shape
                num_groups = data_shape[self.qchannel] // self.group_size
                # Use group-wise quantile computation
                if self.qchannel == 0:
                    data = data.reshape(num_groups, self.group_size, -1)
                    outlier_threshold_upper = torch.quantile(data, t, dim=1, keepdim=True)
                    outlier_threshold_lower = torch.quantile(data, 1-t, dim=1, keepdim=True)
                else:
                    data = data.reshape(-1, num_groups, self.group_size)
                    outlier_threshold_upper = torch.quantile(data, t, dim=-1, keepdim=True)
                    outlier_threshold_lower = torch.quantile(data, 1-t, dim=-1, keepdim=True)
            else:
                # Use original quantile computation
                outlier_threshold_upper = torch.quantile(data, t, dim=self.qchannel).unsqueeze(self.qchannel)
                outlier_threshold_lower = torch.quantile(data, 1-t, dim=self.qchannel).unsqueeze(self.qchannel)
        else:
            #per-token - remove tokenwise outliers and normalize range to [-1,1]
            assert(False) # not currently supported

        # Data is already torch tensor on GPU, no conversion needed

        # range and offset
        rangeval = (outlier_threshold_upper - outlier_threshold_lower) / 2
        zeropoint = (outlier_threshold_upper + outlier_threshold_lower) / 2

        # shift by offset
        data_shifted = data - zeropoint

        # normalize by rangeval into [-1,1]
        data_shifted_normalized = data_shifted / rangeval

        #get outliers (need to mask out for kmeans)
        if not cap_outliers:
            outlier_mask = torch.logical_or((data_shifted_normalized > 1), (data_shifted_normalized < -1))

        # remove first few tokens
        if first_few_fp16 > -1:
            for i in range(0,self.nsamples):
                start = i*self.seqlen
                end = i*self.seqlen + first_few_fp16
                outlier_mask[start:end,:] = True

        if nuq:
            centroids = []
            act_distn_tensor = data_shifted_normalized.flatten()
            n_cluster = 2 ** self.bits

            outlier_mask_unflattened = outlier_mask
            outlier_mask = outlier_mask.flatten()
            act_distn_tensor_without_outliers = act_distn_tensor[~outlier_mask]
            act_distn_tensor_without_outliers = act_distn_tensor_without_outliers.float().reshape(-1, 1)

            # load fisher info
            if fisher is not None:
                fisher_info = fisher.flatten().to(device)  # Move fisher_info to GPU
                fisher_info_tmp_without_outliers = fisher_info[~outlier_mask]
                # Use GPU-accelerated KMeans
                cluster_centers = gpu_kmeans_clustering(
                    act_distn_tensor_without_outliers,
                    n_clusters=n_cluster,
                    sample_weight=fisher_info_tmp_without_outliers,
                    device=device,
                    max_iter=50
                )
            else:
                # Use GPU-accelerated KMeans
                cluster_centers = gpu_kmeans_clustering(
                    act_distn_tensor_without_outliers,
                    n_clusters=n_cluster,
                    device=device,
                    max_iter=50
                )

            centroids.append(cluster_centers)

            #Q-Norm
            if norm and (self.group_size == 0 or self.one_group):
                centroid = torch.tensor(centroids[0], device=device)
                aug = data_shifted_normalized  # Already on GPU
                not_outlier_mask_unflattened = ~outlier_mask_unflattened

                m1 = (aug*not_outlier_mask_unflattened).sum()/not_outlier_mask_unflattened.sum()
                not_outlier_mask_unqueeze = not_outlier_mask_unflattened.sum()
                stdev1 = torch.sqrt(torch.sum(((aug - m1)*not_outlier_mask_unflattened)**2) / not_outlier_mask_unqueeze)

                aug, freq = round_to_nearest_pole_sim(aug, centroid, return_freq=True)

                m2 = (aug*not_outlier_mask_unflattened).sum()/not_outlier_mask_unflattened.sum()
                stdev2 = torch.sqrt(torch.sum(((aug - m2)*not_outlier_mask_unflattened)**2) / not_outlier_mask_unqueeze)

                normscale = (stdev1 / stdev2)
                normoffset = (- m2) * (stdev1 / stdev2) + m1

                return outlier_threshold_upper, outlier_threshold_lower, centroids, normscale, normoffset
            else:
                return outlier_threshold_upper, outlier_threshold_lower, centroids
        else:
            # not using NUQ
            return outlier_threshold_upper, outlier_threshold_lower

    def free(self):
        self.out = None
        self.qout = None
        torch.cuda.empty_cache()

# drop-in layer replacement class
class QuantLinearSim(nn.Module):
    def __init__(
                    self,
                    name,
                    bits,
                    quantizer,
                    infeatures,
                    outfeatures,
                    weight,
                    bias,
                    perchannel=True,
                    include_sparse=False,
                    sparsity_threshold=0.999,
                    dynamicquantization=False,
                    nuq=False,
                    nf_nuq=True,
                    norm=False,
                    first_few_fp16=-1,
                    cap_outliers=-1,
                    clamp=False,
                    group_size=-1,
                    one_group=False
                ):

        super().__init__()
        if bits not in [1,2,3,4,5]:
            raise NotImplementedError("Only 1, 2, 3, 4, 5 bits are supported.")
        self.name = name
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits

        self.weight = weight.T.detach().cpu()
        if bias is not None:
            self.bias = bias.detach().cpu()
        else:
            self.bias = None

        self.perchannel = perchannel
        self.dynamicquantization = dynamicquantization
        self.clamp = clamp

        if perchannel:
            self.qchannel = 0
        else: #per-token quant
            self.qchannel = -1

        self.ochannel = self.qchannel

        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.outlier_threshold_upper = torch.tensor(quantizer[0]).cuda().half()
        self.outlier_threshold_lower = torch.tensor(quantizer[1]).cuda().half()

        self.nuq = nuq
        self.nf_nuq = nf_nuq
        if self.nuq and not self.nf_nuq:
            self.lut = quantizer[2]
        else:
            self.lut = None

        if norm:
            self.normscale = quantizer[3]
            self.normoffset = quantizer[4]
            self.norm = True
        else:
            self.norm = False
            self.normscale = None
            self.normoffset = None

        self.cap_outliers = cap_outliers
        self.first_few_fp16 = first_few_fp16
        
        # Group-wise quantization parameters
        self.group_size = group_size
        self.one_group = one_group

        # for normalfloat support - compute NF signposts
        if self.nf_nuq:
            dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            
            if self.bits == 1:
                # Special case for 1-bit quantization
                # Use simple symmetric signposts
                self.nf_signposts = [-1, 1]
            else:
                # get evenly spaced percentile values
                num_signposts_pos = (2 ** (self.bits - 1)) + 1 # for pos half
                num_signposts_neg = (2 ** (self.bits - 1)) # for neg half

                self.nf_signposts_negative = []
                self.nf_signposts_positive = []

                # from https://arxiv.org/pdf/2306.06965.pdf
                offsets = [0.5*(1/32 + 1/30), 1 - 0.5*(1/32 + 1/30)]
                list1 = [offsets[0]]
                spacing = (0.5 - offsets[0]) / (2 ** (self.bits - 1) - 1)

                add = offsets[0]
                for i in range(num_signposts_neg - 1):
                    add += spacing
                    list1.append(add)

                list2 = []
                spacing = (offsets[1] - 0.5) / (2 ** (self.bits - 1)) #1 extra space
                add = 0.5
                for i in range(num_signposts_pos - 1):
                    list2.append(add)
                    add += spacing
                list2.append(offsets[-1])

                # first do negative part [0->0.5]
                for i in range(num_signposts_neg):
                    v1 = list1[i]
                    val = dist.icdf(torch.tensor([v1])).data.numpy()
                    self.nf_signposts_negative.append(torch.tensor(val).item())

                # next do positive part [0.5->1]
                for i in range(num_signposts_pos):
                    v1 = list2[i]
                    val = dist.icdf(torch.tensor([v1])).data.numpy()
                    self.nf_signposts_positive.append(torch.tensor(val).item())

                signpost_neg_min = self.nf_signposts_negative[0]
                signpost_neg_max = self.nf_signposts_negative[-1]
                rangeval = abs(signpost_neg_min)-abs(signpost_neg_max)
                off = abs(signpost_neg_max)
                for s in range(len(self.nf_signposts_negative)):
                    self.nf_signposts_negative[s] = (self.nf_signposts_negative[s] + off) / rangeval

                signpost_pos_min = self.nf_signposts_positive[0]
                signpost_pos_max = self.nf_signposts_positive[-1]
                rangeval = abs(signpost_pos_max)-abs(signpost_pos_min)
                off = abs(signpost_pos_min)

                for s in range(len(self.nf_signposts_positive)):
                    self.nf_signposts_positive[s] = (self.nf_signposts_positive[s] - off) / rangeval

                del self.nf_signposts_positive[0]

                # delete last negative value and merge
                self.nf_signposts = self.nf_signposts_negative + self.nf_signposts_positive

            assert (len(self.nf_signposts) == (2 ** self.bits))

    #replacement forward pass
    def forward(self, x, other_mat=None):

        out_shape = x.shape[:-1] + (self.outfeatures, )
        x = x.reshape(-1,x.shape[-1])

        # copying weight to / from device during evaluation lets us evaluate
        # a large model with limitted memory usage

        self.weight = self.weight.to(x.device)
        if self.bias is not None:
            self.bias = self.bias.to(x.device)

        x = x.half() # for now cast to fp16 and back (quantization code assumes fp32)
        y = x @ self.weight
        y = y + self.bias if self.bias is not None else y
        y = y.float()

        # if using dense-and-sparse quantization, detect outliers in output tensor
        if self.include_sparse:
            if self.dynamicquantization:
                outlier_mask = get_outliers_dynamic(
                    y,
                    channel=self.ochannel,
                    thresh=self.sparsity_threshold,
                    first_few_fp16=self.first_few_fp16
                )
            else:
                self.outlier_threshold_upper = self.outlier_threshold_upper.to(y.device)
                self.outlier_threshold_lower = self.outlier_threshold_lower.to(y.device)
                outlier_mask = get_outliers(
                    y,
                    channel=self.ochannel,
                    outlier_threshold_upper=self.outlier_threshold_upper,
                    outlier_threshold_lower=self.outlier_threshold_lower,
                    cap_outliers=self.cap_outliers,
                    first_few_fp16=self.first_few_fp16
                )
        else:
            outlier_mask = None

        # quantize output tensor
        if self.nuq:
            if self.nf_nuq:
                y = quant_fn_nf(
                    y,
                    bits=self.bits,
                    qchannel=self.qchannel,
                    maxval=self.outlier_threshold_upper,
                    minval=self.outlier_threshold_lower,
                    include_sparse=self.include_sparse,
                    outlier_mask=outlier_mask,
                    dynamicquantization=self.dynamicquantization,
                    nf_lut=self.nf_signposts,
                    group_size=self.group_size,
                    one_group=self.one_group
                )
            else:
                y = quant_fn_nuq_recon(
                    y,
                    bits=self.bits,
                    qchannel=self.qchannel,
                    maxval=self.outlier_threshold_upper,
                    minval=self.outlier_threshold_lower,
                    include_sparse=self.include_sparse,
                    outlier_mask=outlier_mask,
                    dynamicquantization=self.dynamicquantization,
                    lut=self.lut,
                    norm=self.norm,
                    normscale=self.normscale,
                    normoffset=self.normoffset,
                    first_few_fp16=self.first_few_fp16,
                    group_size=self.group_size,
                    one_group=self.one_group
                )

        else:
            # low-bit uniform simulated quant
            y = quant_fn_zp(
                y,
                bits=self.bits,
                qchannel=self.qchannel,
                maxval=self.outlier_threshold_upper,
                minval=self.outlier_threshold_lower,
                include_sparse=self.include_sparse,
                outlier_mask=outlier_mask,
                dynamicquantization=self.dynamicquantization,
                clamp=self.clamp,
                group_size=self.group_size,
                one_group=self.one_group
            )

        self.weight = self.weight.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

        y = y.reshape(out_shape)

        y = y.half()
        return y

# update modules
def make_quant_sim(
                    module,
                    quantizers,
                    bits,
                    name='',
                    perchannel=True,
                    include_sparse=False,
                    sparsity_threshold=0.999,
                    dynamicquantization=False,
                    nuq=False,
                    nf_nuq=True,
                    norm=False,
                    cap_outliers=-1,
                    first_few_fp16=-1,
                    clamp=False,
                    group_size=-1,
                    one_group=False
                  ):
    if isinstance(module, QuantLinearSim):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in quantizers.keys():
            delattr(module, attr)
            setattr(module, attr, QuantLinearSim(
                                                    name1,
                                                    bits,
                                                    quantizers[name1],
                                                    tmp.in_features,
                                                    tmp.out_features,
                                                    tmp.weight,
                                                    tmp.bias,
                                                    perchannel=perchannel,
                                                    include_sparse=include_sparse,
                                                    sparsity_threshold=sparsity_threshold,
                                                    dynamicquantization=dynamicquantization,
                                                    nuq=nuq,
                                                    nf_nuq=nf_nuq,
                                                    norm=norm,
                                                    cap_outliers=cap_outliers,
                                                    first_few_fp16=first_few_fp16,
                                                    clamp=clamp,
                                                    group_size=group_size,
                                                    one_group=one_group
                                                ))
        del tmp
    for name1, child in module.named_children():
        make_quant_sim(
                        child,
                        quantizers,
                        bits,
                        name + '.' + name1 if name != '' else name1,
                        perchannel=perchannel,
                        include_sparse=include_sparse,
                        sparsity_threshold=sparsity_threshold,
                        dynamicquantization=dynamicquantization,
                        nuq=nuq,
                        nf_nuq=nf_nuq,
                        norm=norm,
                        cap_outliers=cap_outliers,
                        first_few_fp16=first_few_fp16,
                        clamp=clamp,
                        group_size=group_size,
                        one_group=one_group
                      )
