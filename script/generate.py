import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDetermineBonds
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import radius_graph

RDLogger.DisableLog("rdApp.*")

import multiprocessing
import os
import pickle
import time
from math import pi as PI

import numpy as np
from scipy.spatial.transform import Rotation

import utils
from arguments import generate_args_parser
from dataset import DataProcessor
from model import DeepICL


class Generator(DataProcessor):
    def __init__(self, args, model, device):
        super().__init__(args, mode="generate")

        self.model = model
        self.device = device
        self.generate_key_dir = args.key_dir
        self.result_dir = args.result_dir
        self.add_noise = args.add_noise

        self.ligand_coeff = 1.0  # default
        self.pocket_coeff_max = args.pocket_coeff_max
        self.pocket_coeff_thr = args.pocket_coeff_thr
        self.pocket_coeff_beta = args.pocket_coeff_beta

        with open(args.key_dir + "/test_keys.pkl", "rb") as f:
            self.keys = pickle.load(f)

        self.data_list = []
        for k in self.keys:
            with open(args.data_dir + "/" + k, "rb") as f:
                self.data_list.append(pickle.load(f))

        self.max_num_atom = args.max_num_add_atom

        self.use_scaffold = args.use_scaffold
        self.get_traj = False  # TODO

        # Randomness control
        self.translation_coeff = args.translation_coeff
        self.rotation_coeff = args.rotation_coeff
        self.t1 = args.temperature_factor1
        self.t2 = args.temperature_factor2

        self.radial_limits = args.radial_limits
        self.radial_cutoff = args.dist_one_hot_param1[1]
        self.min_dist = args.dist_one_hot_param2[0]
        self.max_dist = args.dist_one_hot_param2[1]
        self.n_bins = args.dist_one_hot_param2[2]

        self.verbose = args.verbose

        self.input_list = self._make_input_list()

    def __len__(self):
        return len(self.keys)

    def run(self, idxs):
        data_idx, sample_idx = idxs

        key = self.keys[data_idx]
        name = f"{key}_{sample_idx}"
        fn = f"{self.result_dir}/{name}.xyz"
        fn2 = f"{self.result_dir}/{name}.sdf"

        if self.verbose:
            print(f"Generating sample: {name}...", flush=True)

        input = self._get_input_from_data(self.data_list[data_idx])
        output = self._generate_molecule(*input)
        msg = f"{name}"
        utils.write_xyz(output[0], output[1], msg=msg, fn=fn, is_onehot=True)
        try:
            mol = AllChem.MolFromXYZFile(fn)
            rdDetermineBonds.DetermineBonds(mol, charge=0)
        except:
            return
        writer = Chem.SDWriter(fn2)
        writer.write(mol)
        writer.close()
        os.unlink(fn)
        return

    def _run(self, idxs):
        ########### DEPRECATED ###########
        #### DELETED IN LATER VERSION ####

        data_idx, sample_idx = idxs

        key = self.keys[data_idx]
        name = f"{key}_{sample_idx}"
        fn = f"{self.result_dir}/{name}.xyz"
        fn2 = f"{self.result_dir}/{name}.sdf"

        if self.verbose:
            print(f"Generating sample: {name}...", flush=True)

        input = self._get_input_from_data(self.data_list[data_idx])
        output = self._generate_molecule(*input)
        msg = f"{name}"
        utils.write_xyz(output[0], output[1], msg=msg, fn=fn, is_onehot=True)
        os.system(f"obabel {fn} -O {fn2} 2> /dev/null")
        os.unlink(fn)
        return

    def _get_random_transformation(self, translation_coeff=0.2, rotation_coeff=PI / 90):
        T = np.random.normal(0, translation_coeff, (1, 3))
        unit_vec = np.random.normal(0, 1, (3,))
        unit_vec /= np.linalg.norm(unit_vec)
        angle = np.random.normal(0, rotation_coeff, (1,))
        rot_vec = angle * unit_vec
        R = Rotation.from_rotvec(rot_vec).as_matrix()
        return torch.Tensor(T), torch.Tensor(R)

    def _pocket_coeff_scheduler(
        self,
        mean_dist,
    ):
        self.pocket_coeff = self.pocket_coeff_max * torch.exp(
            (self.pocket_coeff_thr - mean_dist) * self.pocket_coeff_beta
        )
        self.pocket_coeff = torch.clamp(self.pocket_coeff, max=self.pocket_coeff_max)

    def _make_input_list(
        self,
    ):
        input_list = [
            (i, j) for i in range(len(self.keys)) for j in range(self.args.num_sample)
        ]
        return input_list

    @torch.no_grad()
    def _get_input_from_data(self, data):
        r"""
        Over-writing original function
        """
        # Get data generated from data/pdbbind.py
        (
            ligand_type,
            ligand_coord,
            pocket_type,
            pocket_coord,
            _,
            ligand_prop,
            scaff_prop,
            _,
            scaff_n,
            _,
            _,
            pocket_prop,
            center_of_mass
        ) = data

        # Pocket info
        p_dict = dict()
        p_Z = torch.Tensor(pocket_type)
        p_R = torch.Tensor(pocket_coord)
        p_c = torch.Tensor(pocket_prop)
        if self.add_noise:
            T, R = self._get_random_transformation(
                self.translation_coeff, self.rotation_coeff
            )
            p_R = p_R @ R + T
            p_dict.update({"translation": T, "rotation": R})
        p_n = p_Z.shape[0]
        com = torch.Tensor(center_of_mass)
        p_dict.update(
            {
                "p_Z": p_Z, 
                "p_R": p_R, 
                "p_n": p_n, 
                "p_cond": p_c,
                "com": com
            }
        )

        if self.use_scaffold:
            s_Z = torch.Tensor(ligand_type)[:scaff_n]
            s_R = torch.Tensor(ligand_coord)[:scaff_n]
            s_n = scaff_n
            s_dict = {
                "s_Z": s_Z,
                "s_R": s_R,
                "s_n": s_n,
                "prop": (ligand_prop, scaff_prop),
            }

            return p_dict, s_dict

        else:
            return p_dict, None

    @torch.no_grad()
    def _generate_molecule(
        self,
        p_dict,
        s_dict=None,
    ):
        r""" """

        latent = torch.randn((1, self.args.num_latent_feature), device=self.device)

        if not self.args.conditional:
            p_cond = None
        elif self.args.use_condition:
            p_cond = p_dict["p_cond"]
        else:  # conditional=True but use_condition=False
            # Blank conditioning
            blank = torch.eye(utils.NUM_INTERACTION_TYPES)[-1].unsqueeze(0)
            p_cond = blank.repeat(p_dict["p_n"], 1)

        i = 0
        l_Z_init = self.o_Z
        l_R_init = self.o_R

        if self.use_scaffold:
            s_n = s_dict["s_n"]
            self.max_num_atom += s_n
            avail = p_dict["p_Z"].new_zeros((self.max_num_atom,)).float()
            for j in range(s_n):
                avail[j] = 1.0
            i = s_n
            l_Z_init = torch.cat([l_Z_init, s_dict["s_Z"]], 0)
            l_R_init = torch.cat([l_R_init, s_dict["s_R"]], 0)
        else:
            avail = p_dict["p_Z"].new_zeros((self.max_num_atom,)).float()

        # New mol, includes the center-of-mass token + scaffold atom
        l_Z_final = l_Z_init[1:]
        l_R_final = l_R_init[1:]

        traj = [(l_Z_final, l_R_final)]

        while i < self.max_num_atom:
            if i > 0 and torch.sum(avail) == 0:
                break
            if i == 0:
                f_R = self.o_R
            else:
                now = torch.multinomial(avail, 1)[
                    0
                ]  # Choose current index within available nodes
                f_R = l_R_init[now + 1 : now + 2]

            # Add token
            l_Z_in = torch.cat([self.f_Z, l_Z_init], 0)
            l_R_in = torch.cat([f_R, l_R_init], 0)

            # Get surrounding pocket atoms
            p_ind = torch.topk(
                torch.cdist(f_R, p_dict["p_R"]), k=self.args.k, dim=-1, largest=False
            )[1][0]
            p_Z_in = p_dict["p_Z"][p_ind]
            p_R_in = p_dict["p_R"][p_ind]
            p_to_f_dist = torch.mean(torch.cdist(f_R, p_R_in))
            if self.args.conditional:
                p_c = p_cond[p_ind]
            else:
                p_c = None

            # Make PyG HeteroData
            data = HeteroData()
            data["ligand"].x = l_Z_in
            data["pocket"].x = p_Z_in
            data["ligand"].pos = l_R_in
            data["pocket"].pos = p_R_in

            l2l_index = radius_graph(l_R_in, r=self.radial_cutoff)
            p2p_index = radius_graph(p_R_in, r=self.radial_cutoff)
            l_index = (
                torch.LongTensor(list(range(i + self.num_token)))
                .unsqueeze(1)
                .repeat(1, self.args.k)
                .view(-1)
            )
            p_index = (
                torch.LongTensor(list(range(self.args.k)))
                .unsqueeze(0)
                .repeat(i + self.num_token, 1)
                .view(-1)
            )

            data["ligand", "l2l", "ligand"].edge_index = l2l_index
            data["pocket", "p2p", "pocket"].edge_index = p2p_index
            data["pocket", "p2l", "ligand"].edge_index = torch.stack(
                [p_index, l_index], dim=0
            )

            l2l_weight = torch.norm(
                data["ligand"].pos[data["l2l"].edge_index[0]]
                - data["ligand"].pos[data["l2l"].edge_index[1]],
                dim=-1,
            )
            p2p_weight = torch.norm(
                data["pocket"].pos[data["p2p"].edge_index[0]]
                - data["pocket"].pos[data["p2p"].edge_index[1]],
                dim=-1,
            )
            p2l_weight = torch.norm(
                data["pocket"].pos[p_index] - data["ligand"].pos[l_index], dim=-1
            )

            l2l_attr = self.distance_expand1(l2l_weight)[0]
            p2p_attr = self.distance_expand1(p2p_weight)[0]
            p2l_attr = self.distance_expand1(p2l_weight)[0]

            data["l2l"].edge_weight = l2l_weight
            data["p2p"].edge_weight = p2p_weight
            data["p2l"].edge_weight = p2l_weight

            data["l2l"].edge_attr = l2l_attr
            data["p2p"].edge_attr = p2p_attr
            data["p2l"].edge_attr = p2l_attr

            data = Batch.from_data_list([data])

            # Embed(propagate) unfinished graph
            self.model.embedding(data, cond=p_c)
            self._pocket_coeff_scheduler(p_to_f_dist)

            if self.args.conditional:
                data["pocket"].h = self.model.latent_mlp(
                    torch.cat(
                        [
                            data["pocket"].h,
                            p_c,
                            latent.repeat(data["pocket"].h.shape[0], 1),
                        ],
                        -1,
                    )
                )
            else:
                data["pocket"].h = self.model.latent_mlp(
                    torch.cat(
                        [data["pocket"].h, latent.repeat(data["pocket"].h.shape[0], 1)],
                        -1,
                    )
                )

            next_type_ll_log_prob = self.ligand_coeff * self.model.next_type_ll(
                data, "ligand"
            ).squeeze(0)
            next_type_lp_log_prob = self.pocket_coeff * self.model.next_type_lp(
                data, "pocket"
            ).squeeze(0)
            next_type_log_prob = next_type_ll_log_prob + next_type_lp_log_prob
            next_type_log_prob -= torch.logsumexp(
                next_type_log_prob, dim=-1, keepdim=True
            )
            if self.t1 == 0:
                next_type_ind = torch.argmax(next_type_log_prob)
            else:
                next_type_log_prob /= self.t1
                next_type_log_prob -= torch.logsumexp(
                    next_type_log_prob, dim=-1, keepdim=True
                )
                next_type_ind = torch.multinomial(next_type_log_prob.exp(), 1)[0]


            if next_type_ind == self.args.num_ligand_atom_feature - 1:  # Termination
                avail[now] = 0.0
                continue

            next_type = torch.eye(self.args.num_ligand_atom_feature)[next_type_ind]
            next_type = next_type.unsqueeze(0)

            if i == 0:
                grid = self._make_grid(f_R, (0.0, self.radial_limits[-1]))  # TODO
            else:
                grid = self._make_grid(f_R, self.radial_limits)

            next_dist_ll_log_prob = self.ligand_coeff * self.model.next_dist_ll(
                data, next_type, "ligand"
            )
            next_dist_lp_log_prob = self.pocket_coeff * self.model.next_dist_lp(
                data, next_type, "pocket"
            )
            next_dist_log_prob = torch.cat(
                [next_dist_lp_log_prob, next_dist_ll_log_prob], 0
            )
            next_dist_log_prob -= torch.logsumexp(
                next_dist_log_prob, dim=-1, keepdim=True
            )

            dists = torch.cdist(torch.cat([p_R_in, l_R_in], 0), grid)
            dists -= self.min_dist
            dists *= (self.n_bins - 1) / (self.max_dist - self.min_dist)
            dists.clamp_(0, self.n_bins - 1)
            dist_inds = dists.long()
            grid_log_prob = torch.gather(next_dist_log_prob, -1, dist_inds)
            grid_log_prob = grid_log_prob.sum(0)
            grid_log_prob -= torch.logsumexp(grid_log_prob, -1, keepdim=True)
            if self.t2 == 0:
                next_coord_ind = torch.argmax(grid_log_prob)
            else:
                grid_log_prob /= self.t2
                grid_log_prob -= torch.logsumexp(grid_log_prob, -1, keepdim=True)
                next_coord_ind = torch.multinomial(grid_log_prob.exp(), 1)[0]


            next_coord = grid[next_coord_ind]
            next_coord = next_coord.unsqueeze(0)

            l_Z_init = torch.cat([l_Z_init, next_type], 0)
            l_R_init = torch.cat([l_R_init, next_coord], 0)

            l_Z_final = l_Z_init[1:]  # Remove origin token
            l_R_final = l_R_init[1:]  # Remove origin token
            traj += [(l_Z_final, l_R_final)]

            avail[i] = 1.0
            i += 1


        if self.add_noise:
            l_R_final = (l_R_final - p_dict["translation"]) @ torch.t(
                p_dict["rotation"]
            )
        l_R_final += p_dict["com"]

        if self.get_traj:
            # Make traj file
            for i, (z, r) in enumerate(traj):
                utils.write_xyz(z, r, msg=f"{i}-th trajectory", fn=f"traj/traj_{i}.xyz")
                utils.xyz_to_sdf(f"traj/traj_{i}.xyz", f"traj/traj_{i}.sdf")
                os.unlink(f"traj/traj_{i}.xyz")

        return l_Z_final, l_R_final, torch.sum(avail) == 0

    @torch.no_grad()
    def _make_grid(
        self,
        center,
        radial_limits,
    ):
        r"""
        [Code adjusted from the original code of G-SchNet]
        Get grid coordinates within the radial limit

        Args:
            center (torch.Tensor): [num_dimension]
            radial_limits (list of float): [min_dist, max_dist]

        Returns:
            grid (torch.Tensor): [num_grid_coords num_dimension]
        """
        num_dimension = self.num_dim  # make grid in 3d space
        grid_max = radial_limits[1]
        grid_steps = int(grid_max * 2 * self.n_bins / self.max_dist)
        coords = np.linspace(-grid_max, grid_max, grid_steps)
        grid = np.meshgrid(*[coords for _ in range(num_dimension)])
        grid = np.stack(grid, axis=-1)  # stack to array (instead of list)

        # reshape into 2d array of positions
        shape_a0 = np.prod(grid.shape[:num_dimension])
        grid = np.reshape(grid, (shape_a0, -1))
        # cut off cells that are out of the spherical limits
        grid_dists = np.sqrt(np.sum(grid**2, axis=-1))
        grid_mask = np.logical_and(
            grid_dists >= radial_limits[0], grid_dists <= radial_limits[1]
        )
        grid = grid[grid_mask]
        grid = torch.Tensor(grid, device=center.device)

        # re-center the grid
        grid = grid + center
        return grid


def main():
    # 0. Argument setting
    args = generate_args_parser()
    d = vars(args)
    lines = [f"{a} = {d[a]}\n" for a in d]
    if args.verbose:
        print("####################################################")
        for line in lines:
            print(line, end="")
        print("####################################################")

    # 1. Device setting
    multiprocessing.set_start_method("spawn")
    if args.ncpu > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
        device = torch.device("cpu")
    elif args.ngpu > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = utils.get_cuda_visible_devices(args.ngpu)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if args.verbose:
        print("Device:", device)

    # 2. Model setting
    model = DeepICL(args)
    model = utils.initialize_model(model, args.ngpu > 0, args.restart_dir)
    generator = Generator(args, model, device)

    # 3. Directory setting
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if len(os.listdir(args.result_dir)) > 0:
        if args.y:
            os.system(f"rm {args.result_dir}/*")
        else:
            token = input(f"Remove existing files in {args.result_dir}?: (y/n)")
            if token == "y":
                os.system(f"rm {args.result_dir}/*")
            elif token == "n":
                pass
            else:
                print("Wrong input:", token, "--> Exiting...")
                exit()
    if args.verbose:
        print("Saving results in:", args.result_dir)
    with open(f"{args.result_dir}/generation_params.log", "w") as w:
        w.writelines(lines)

    # 4. Generation
    if args.ncpu > 1:
        st = time.time()
        pool = multiprocessing.Pool(args.ncpu)
        r = pool.map_async(generator._run, generator.input_list)
        r.wait()
        pool.close()
        pool.join()
        if args.verbose:
            delta_t = time.time() - st
            N = args.num_sample * len(generator.keys)
            print(f"Duration: {delta_t:.2f} (s)")
            print(f"Sampling speed: {delta_t / N:.2f} (s/sample)")
    else:
        for x in generator.input_list:
            generator._run(x)


if __name__ == "__main__":
    main()
