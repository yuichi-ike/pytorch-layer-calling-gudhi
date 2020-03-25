import torch
import torch.nn as nn
import collections
import numpy as np
from numpy import isinf
import gudhi as gd
from scipy.spatial import distance as sci_distance
from .util.common import st_StructureW, DTM_values





class DtmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, max_dim, max_edge_len, kNN):
        """
        x: a point cloud
        ------------------
        Parameters:
            hom_dim : list of intergers
                specify homological dimension to compute
            max_edge_len : float
                maximal edge length for calculation
        """

        device = torch.device('cpu')
        

        print('ripsLayer:forward: x')
        print(x)

        x_np = x.detach().numpy().copy() #make a numpy copy of the input tensor
        dimension = x.shape[1]

        DTM_val, NN_Dist, NN = DTM_values(x_np,x_np,kNN)
        distances = sci_distance.pdist(x_np)
        distances = sci_distance.squareform(distances)

        dimension_max = max_dim + 1

        simplex_tree = st_StructureW(x_np, DTM_val, distances, edge_max=max_edge_len, dimension_max=dimension_max)
        simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)

        P = simplex_tree.persistence_pairs() #pairs of simplices associated with birth and death of points in the PD.
                                                           #note this array is not alligned with the array of (birth,death) pairs computed by persistence 
        filtration = simplex_tree.filtration
        sqrt_ = np.sqrt(np.sum(NN_Dist*NN_Dist, axis=1) * kNN)

        import pdb; pdb.set_trace()

        #print('ripsLayer:forward:persistence_pairs')
        #print(persistence_pairs)

        diagram = torch.zeros(2*len(P)) 
        derivative = torch.zeros((2*len(P), dimension*len(x_np)))


        for (num, (birth_simp, death_simp)) in enumerate(P):
            # print(birth_simp,death_simp)

            hom_degree = len(birth_simp)-1

            birth_time = filtration(birth_simp)
            death_time = filtration(death_simp)

            if death_time == np.inf:
                death_time = max_edge_length
                
            
            persistence_diagram.append(
                (hom_degree, (birth_time, death_time)))
            dw_pd[hom_degree].append((birth_time, death_time))

            birth_max_vertices = ind_max_filtr(birth_simp, filtration)
            death_max_vertices = ind_max_filtr(death_simp, filtration)

            # print(birth_max_vertices,death_max_vertices)
            diagram_info.append((birth_max_vertices, death_max_vertices))
    
            bmv0 = birth_max_vertices[0]
            bmv1 = birth_max_vertices[1]

            if bmv0 is not None and bmv1 is None and knn != 1:
                bpt0 = pc[bmv0]

                bNN0 = NN[bmv0, :]

                denominator0 = sqrt_[bmv0]

                if denominator0 != 0:
                    for bnn0 in bNN0[1:knn]:
                        bnpt0 = pc[bnn0]
                        derivative[2*num, 3*bmv0:3*(bmv0+1)] += 1/denominator0 * (bpt0-bnpt0)
                        derivative[2*num, 3*bnn0:3*(bnn0+1)] += 1/denominator0 * (bnpt0-bpt0)

            if bmv0 is not None and bmv1 is not None:
                bpt0 = pc[bmv0]
                bpt1 = pc[bmv1]

                distance_ = distances[bmv1, bmv0]
                #print("distance_", distance_)

                if distance_ != 0:
                    # 0.5*1 = 0.5
                    derivative[2*num, 3*bmv0:3*(bmv0+1)] = 0.5/distance_ * (bpt0-bpt1)
                    derivative[2*num, 3*bmv1:3*(bmv1+1)] = 0.5/distance_ * (bpt1-bpt0)

                if knn != 1:
                    bNN0 = NN[bmv0, :]
                    bNN1 = NN[bmv1, :]

                    denominator0 = sqrt_[bmv0]
                    denominator1 = sqrt_[bmv1]

                    for bnn0, bnn1 in zip(bNN0[1:knn], bNN1[1:knn]):
                        bnpt0 = pc[bnn0]
                        bnpt1 = pc[bnn1]
                        if denominator0 != 0:
                            derivative[2*num, 3*bmv0:3*(bmv0+1)] += 0.5/denominator0 * (bpt0-bnpt0)
                            derivative[2*num, 3*bnn0:3*(bnn0+1)] += 0.5/denominator0 * (bnpt0-bpt0)
                        if denominator1 != 0:
                            derivative[2*num, 3*bmv1:3*(bmv1+1)] += 0.5/denominator1 * (bpt1-bnpt1)
                            derivative[2*num, 3*bnn1:3*(bnn1+1)] += 0.5/denominator1 * (bnpt1-bpt1)

            dmv0 = death_max_vertices[0]
            dmv1 = death_max_vertices[1]

            if dmv0 is not None and dmv1 is not None:
                #print("death_max_vertices", death_max_vertices)

                dpt0 = pc[dmv0]
                dpt1 = pc[dmv1]

                #print("filtration", simplex_tree.filtration(death_simp))
                #print("norm", np.linalg.norm(dpt0-dpt1))
                # print(distances[dmv1][dmv0])

                distance_ = distances[dmv1, dmv0]
                #print("distance_", distance_, distance_ == 0)

                if distance_ != 0:
                    # 0.5*1 = 0.5
                    derivative[2*num+1, 3*dmv0:3*(dmv0+1)] += 0.5/distance_ * (dpt0-dpt1)
                    derivative[2*num+1, 3*dmv1:3*(dmv1+1)] += 0.5/distance_ * (dpt1-dpt0)

                if knn != 1:
                    #print(NN[dmv0, :], NN[dmv1])
                    dNN0 = NN[dmv0, :]
                    dNN1 = NN[dmv1, :]

                    denominator0 = sqrt_[dmv0]
                    denominator1 = sqrt_[dmv1]

                    for dnn0, dnn1 in zip(dNN0[1:knn], dNN1[1:knn]):
                        dnpt0 = pc[dnn0]
                        dnpt1 = pc[dnn1]
                        if denominator0 != 0:
                            derivative[2*num+1, 3*dmv0:3*(dmv0+1)] += 0.5/denominator0 * (dpt0-dnpt0)
                            derivative[2*num+1, 3*dnn0:3*(dnn0+1)] += 0.5/denominator0 * (dnpt0-dpt0)
                        if denominator1 != 0:
                            derivative[2*num+1, 3*dmv1:3*(dmv1+1)] += 0.5/denominator1 * (dpt1-dnpt1)
                            derivative[2*num+1, 3*dnn1:3*(dnn1+1)] += 0.5/denominator1 * (dnpt1-dpt1)

        

        #print('ripsLayer:forward:derMatTensor')
        #print(derMatTensor)

        ctx.derMatTensor = derivative 

        #import pdb;pdb.set_trace()
        return diagram

    @staticmethod 
    def backward(ctx, gradOp):

        #import pdb;pdb.set_trace()
        #print('ripsLayer:backward:gradOp')
        #print(gradOp)

        #print('ripsLayer:backward:derMatTensor')
        #print(ctx.derMatTensor)

        #print('ripsLayer:backward:product')
        #print(torch.mv(ctx.derMatTensor,gradOp))

        return torch.matmul(ctx.derMatTensor,gradOp), None, None 

class DtmFiltration(nn.Module):
    def __init__(self, hom_dim, max_edge_len, kNN):
        super(DtmFiltration, self).__init__()
        self.hom_dim = hom_dim
        self.max_edge_len = max_edge_len
        self.kNN = kNN

    def forward(self, x):
        return DtmFunction.apply(x,self.hom_dim, self.max_edge_len, self.kNN)


