import torch
import torch.nn as nn

from .net import *
from .model_util import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DGU_Block(nn.Module):
    def __init__(self):
        super(DGU_Block, self).__init__()
        
        print("Loading Enhance Net...")
        Proximal_t = [RDN(3,1)]
        self.Proximal_t = nn.Sequential(*Proximal_t)
        
        self.t_1D_Net = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=False))

        self.rho_1 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.rho_2 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.rho_3 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.rho_4 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.rho_5 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.rho_6 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.rho_7 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)


    def forward(self, I, t_p, B_p, B, t, J, Aux_J, Aux_t, Lag_J, Lag_t, Map_u, Map_v, u, v, w, labels, det_api, patch_size = 35, eps = 1e-6):     
        # === Paper Note ===
        # The hyperparameter(eta) values reported in the Experimental Settings section of the IEEE Access version are partially mis-typed.
        # The values used here correspond to the actual configuration used during training and evaluation.
        # eta_0 = 1.0
        # eta_1 = 0.3
        # eta_2 = 0.7
        # eta_3 = 1.0
        # eta_4 = 1.0
        # eta_7 = 1.0
        eta_0 = 1.0
        eta_1 = 1.0
        eta_2 = 1.0
        eta_3 = 0.3
        eta_4 = 0.7
        eta_5 = 1.0
        rho_1 = self.rho_1
        rho_2 = self.rho_2
        rho_3 = self.rho_3
        rho_4 = self.rho_4
        rho_5 = self.rho_5
        rho_6 = self.rho_6
        rho_7 = self.rho_7
           
        sorted_channel_means, sorted_channel_indices, J_l, J_m, J_s, large_channel_idx, medium_channel_idx, small_channel_idx = extract_channel_statistics(J)
        J_l_bar = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(sorted_channel_means[:,2],1),1),1).to(DEVICE)
        J_m_bar = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(sorted_channel_means[:,1],1),1),1).to(DEVICE)
        J_s_bar = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(sorted_channel_means[:,0],1),1),1).to(DEVICE)
        
        J_l = J_l.to(DEVICE)
        J_m = J_m.to(DEVICE)
        J_s = J_s.to(DEVICE)
        
        J_m = J_m + torch.mul(J_l_bar - J_m_bar, J_l)
        J_s = J_s + torch.mul(J_l_bar - J_s_bar, J_l)
        
        J = replace_channel_by_index(J.clone(), J_m.clone(), medium_channel_idx)
        J = replace_channel_by_index(J.clone(), J_s.clone(), small_channel_idx)

        
        D = torch.ones(I.shape).to(DEVICE) 
        B = (eta_4*B_p - eta_0*(J*t - I)*(1-t))/(eta_0*(1.0 - t)*(1.0 - t) + eta_4 )
        B = torch.mean(B, (2,3), True)  
        B = B*D
        
        t = (eta_3*t_p + rho_4*Aux_t - Lag_t - eta_0*(B - I)*(J - B))/(eta_0*(J - B)*(J - B) + eta_3 + rho_4)
        t = self.t_1D_Net(t)
        t = torch.cat((t, t, t), 1)  
        
        J = (eta_0*(t*(I-B*(1.0 - t)) + rho_3*Aux_J - Lag_J + rho_5*u - rho_6*v + rho_7*w + rho_6))/(eta_0*t*t + rho_3 + rho_5 + rho_6 + rho_7)

        M_T_P = u
        M_T_Q = v
        
        u = (rho_1*M_T_P + rho_5*J)/(rho_1 + rho_5)
        v = (rho_2*M_T_Q - rho_6*J + rho_6)/(rho_2 + rho_6)
        
        Aux_t = self.Proximal_t(t + (1.0/rho_4)*Lag_t)
        
        Lag_J = Lag_J + rho_3*(J - Aux_J)
        Lag_t = Lag_t + rho_4*(t - Aux_t)
        
        M_u, index_map_dark = get_dark_channel(u, patch_size)
        M_v, index_map_dark = get_dark_channel(v, patch_size)
        
        Map_u = softThresh(M_u, eta_1/rho_1)
        Map_v = softThresh(M_v, eta_2/rho_2)
        
        ### Detection-guided prior
        with torch.no_grad():
            J_input = J.detach().clamp(0, 1) 
            bs, C, H, W = J_input.shape
            results = det_api.predict(J_input, imgsz=640, verbose=False)
            
            diag_dict = build_classwise_L_det(results, J_input, num_classes=det_api.model.nc)
            M_gt_dict = build_classwise_M_gt(labels, J_input, num_classes=det_api.model.nc)


        w_batch = torch.empty_like(J)  
        eps_den = 1e-12

        for b in range(bs):
            J_vec = J[b].reshape(-1)         
            denom = rho_7 + torch.zeros_like(J_vec)   
            num   = rho_7 * J_vec                 

            for k in range(det_api.model.nc):
                key = (b, k)
                l_det_diag = diag_dict.get(key, None)
                m_gt_vec   = M_gt_dict.get(key, None)
                if (l_det_diag is None) or (m_gt_vec is None):
                    continue

                l_det_diag = l_det_diag.to(J_vec.device, dtype=J_vec.dtype)
                m_gt_vec   = m_gt_vec.to(J_vec.device,   dtype=J_vec.dtype)

                denom = denom + eta_5 * (l_det_diag * l_det_diag)
                num   = num   + eta_5 * (l_det_diag * m_gt_vec)

            w_batch[b] = (num / (denom + eps_den)).view(C, H, W)

        w = w_batch  
        
        return B, t, J, Aux_J, Aux_t, Lag_J, Lag_t, Map_u, Map_v , u, v, w, rho_3
        
        

        
class DGU_Net(nn.Module):
    def __init__(self, Block_number=5):
        super(DGU_Net, self).__init__()
        self.Block_number = Block_number
        block_list = []
        for i in range(self.Block_number):
            block_list.append(DGU_Block())
        self.enhance_net = nn.ModuleList(block_list)
        
        n_feat=40; scale_unetfeats=20; kernel_size=3; reduction=4; bias=False
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4)
        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.merge12=mergeblock(n_feat,3,True)
        
        self.Proximal_J = IPMM(in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False)


    def forward(self, I, t_p, B_p, labels, det_api):
        bs, _, _, _ = I.shape
        B = torch.zeros((bs, 3, 1, 1)).to(DEVICE) 
        t = torch.zeros(I.shape).to(DEVICE) 
        J = I.to(DEVICE) 
        
        
        Aux_J = torch.zeros(I.shape).to(DEVICE) 
        Aux_t = torch.zeros(I.shape).to(DEVICE)
        
        Lag_J = torch.zeros(I.shape).to(DEVICE)         
        Lag_t = torch.zeros(I.shape).to(DEVICE)      
        
        Map_u = torch.zeros(I.shape).to(DEVICE)
        Map_v = torch.zeros(I.shape).to(DEVICE)
        
        u = torch.zeros(I.shape).to(DEVICE)  
        v = torch.zeros(I.shape).to(DEVICE)
        
        w = torch.zeros(I.shape).to(DEVICE)
        
        list_J = []
        list_t = []
        list_B = []
        
        rho_3 = torch.tensor([3.001]).to(DEVICE)
        
        x1_img = J +(1.0/rho_3) * Lag_J
        x1 = self.shallow_feat1(x1_img)
        feat1, feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1, feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], J)
        
        Aux_J = stage1_img

        
        for j in range(self.Block_number):
            [B, t, J, Aux_J, Aux_t, Lag_J, Lag_t, Map_u, Map_v, u, v, w, rho_3] = self.enhance_net[j](I, t_p, B_p, B, t, J, Aux_J, Aux_t, Lag_J, Lag_t, Map_u, Map_v, u, v, w, labels, det_api)
            
            img = J + (1.0/rho_3)*Lag_t
            x2_samfeats, stage1_img, feat1, res1 = self.Proximal_J(img, stage1_img,feat1,res1,x2_samfeats)
            Aux_J = stage1_img            

            list_J.append(J)
            list_t.append(t)
            list_B.append(B)
        
        return list_J, list_t, list_B