"""The descriptive NumPy KL matches the actual PPO reference objective."""
import numpy as np
import pytest
import torch
from scripts.report_warehouse_v642_reference_drift import reference_drift
from backend.training.warehouse_v622_actor_training import masked_reference_kl


def test_reference_direction_mask_and_float64_match_actual_PPO():
    old=np.array([[1000.,2.,3.,-9.,0.],[2.,-999.,1.,4.,3.]],np.float32)
    new=np.array([[-1000.,1.,2.,-6.,.1],[1.,999.,1.,5.,2.]],np.float32)
    mask=np.array([[0.,1.,1.,1.,1.],[1.,0.,0.,1.,1.]],np.float32)
    kl,same=reference_drift(old,new,mask)
    expected=masked_reference_kl(torch.from_numpy(new),torch.from_numpy(old),torch.from_numpy(mask)).numpy()
    np.testing.assert_allclose(kl,expected,atol=1e-14,rtol=1e-14)
    assert same.tolist()==[True,True]
    zero,agree=reference_drift(old,old,mask);assert np.array_equal(zero,np.zeros(2)) and agree.all()


def test_nonfinite_logits_and_illegal_wait_are_rejected():
    old=np.zeros((1,5));mask=np.ones((1,5));mask[0,4]=0
    with pytest.raises(RuntimeError):reference_drift(old,old,mask)
    mask[0,4]=1;old[0,0]=np.nan
    with pytest.raises(RuntimeError):reference_drift(old,old,mask)
