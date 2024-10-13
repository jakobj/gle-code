import torch


class GLELinear(torch.nn.Linear):

    def compute_error(self, e):
        dtype_original = e.dtype
        res = torch.mm(e.to(self.weight.dtype), self.weight)
        return res.to(dtype_original)

    def compute_grad(self, r_bottom, e):
        grad_weight = - torch.bmm(e.unsqueeze(2), r_bottom.unsqueeze(1)).mean(0)
        self.weight.grad = grad_weight.to(self.weight.dtype)
        assert torch.all(torch.isfinite(self.weight.grad)), breakpoint()
        if self.bias is not None:
            grad_bias = - e.mean(0)
            self.bias.grad = grad_bias.to(self.bias.dtype)
            assert torch.all(torch.isfinite(self.bias.grad)), breakpoint()

    def forward(self, x):
        dtype_original = x.dtype
        res = super().forward(x.to(self.weight.dtype))
        return res.to(dtype_original)
