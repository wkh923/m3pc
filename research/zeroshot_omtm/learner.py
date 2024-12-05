from collections import defaultdict
from typing import Any, Dict

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from research.finetune_omtm.model import *
from research.omtm.models.mtm_model import omtm
from research.omtm.tokenizers.base import TokenizerManager
from research.zeroshot_omtm.masks import *


class Learner(object):
    def __init__(
        self,
        cfg,
        env: gym.Env,
        data_shapes,
        model_config,
        pretrain_model_path,
        obs_mean,
        obs_std,
        tokenizer_manager: TokenizerManager,
        discrete_map: Dict[str, torch.Tensor],
    ):
        self.cfg = cfg
        self.env = env
        self.mtm = model_config.create(data_shapes, cfg.traj_length, discrete_map)
        self.mtm.load_state_dict(
            torch.load(pretrain_model_path, map_location="cpu")["model"]
        )
        self.mtm.to(cfg.device)

        self.tokenizer_manager = tokenizer_manager
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.discrete_map = discrete_map
        self.mtm_optimizer = omtm.configure_optimizers(
            self.mtm,
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999),
        )
        self.eps = 1e-5
        self.action_list = []

        def _schedule(step):
            return 0.5 * (1 + np.cos(step / cfg.num_train_steps * np.pi))

        self.mtm_scheduler = LambdaLR(self.mtm_optimizer, lr_lambda=_schedule)
        self.temp_optimizer = torch.optim.Adam(
            [self.mtm.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

    @torch.no_grad()
    def action_id_sample(
        self,
        sequence_history,
        percentage=1.0,
        horizon=4,
        plan=True,
        eval=False,
        rtg=None,
    ):
        if eval == True:
            assert rtg is not None

        horizon = self.cfg.horizon
        end_idx = sequence_history["path_length"]
        if end_idx + horizon < self.cfg.traj_length:
            horizon = self.cfg.traj_length - end_idx
        smart_traj_length = self.cfg.traj_length
        if end_idx + horizon > 1000:
            smart_traj_length = smart_traj_length - (end_idx + horizon - 1000)
        obs_dim = sequence_history["observations"].shape[-1]
        action_dim = sequence_history["actions"].shape[-1]
        zero_trajectory = {
            "observations": np.zeros((1, self.cfg.traj_length, obs_dim)),
            "actions": np.zeros((1, self.cfg.traj_length, action_dim)),
            "rewards": np.zeros((1, self.cfg.traj_length, 1)),
            "values": np.zeros((1, self.cfg.traj_length, 1)),
        }
        history_length = self.cfg.traj_length - horizon + 1

        for k in zero_trajectory.keys():
            zero_trajectory[k][0, :history_length] = sequence_history[k][
                end_idx - history_length + 1 : end_idx + 1
            ]

        # but obs can be all traj long

        zero_trajectory["observations"][0, :smart_traj_length] = sequence_history[
            "observations"
        ][
            end_idx
            - history_length
            + 1 : end_idx
            - history_length
            + 1
            + self.cfg.traj_length
        ]

        torch_zero_trajectory = {
            (
                "states" if k == "observations" else "returns" if k == "values" else k
            ): torch.tensor(v, device=self.cfg.device, dtype=torch.float32)
            for k, v in zero_trajectory.items()
        }

        if rtg is not None:
            # Use time-related rtg for eval
            return_to_go = float(rtg)
            returns = return_to_go * np.ones((1, self.cfg.traj_length, 1))
            torch_zero_trajectory["returns"] = torch.from_numpy(returns).to(
                self.cfg.device
            )
        else:
            # Use a constant rtg for explore
            return_max = self.tokenizer_manager.tokenizers["returns"].stats.max
            return_min = self.tokenizer_manager.tokenizers["returns"].stats.min

            return_value = return_min + (return_max - return_min) * percentage
            return_to_go = float(return_value)
            returns = return_to_go * np.ones((1, self.cfg.traj_length, 1))
            torch_zero_trajectory["returns"] = torch.from_numpy(returns).to(
                self.cfg.device
            )

        idbc_mask = create_gid_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - horizon
        )
        encode = self.tokenizer_manager.encode(torch_zero_trajectory)
        action_dist = self.tokenizer_manager.decode(self.mtm(encode, idbc_mask))[
            "actions"
        ]  # dist of shape(1, seq_len, act_dim)

        sample_action = action_dist.sample()[0, self.cfg.traj_length - horizon]

        eval_action = action_dist.mean[0, self.cfg.traj_length - horizon]

        if eval:
            return eval_action
        else:
            return sample_action

    @torch.no_grad()
    def action_piid_sample(
        self,
        sequence_history,
        percentage=1.0,
        horizon=4,
        plan=True,
        eval=False,
        rtg=None,
    ):
        if eval == True:
            assert rtg is not None

        horizon = self.cfg.horizon
        end_idx = sequence_history["path_length"]
        if end_idx + horizon < self.cfg.traj_length:
            horizon = self.cfg.traj_length - end_idx
        smart_traj_length = self.cfg.traj_length
        if end_idx + horizon > 1000:
            smart_traj_length = smart_traj_length - (end_idx + horizon - 1000)
        obs_dim = sequence_history["observations"].shape[-1]
        action_dim = sequence_history["actions"].shape[-1]
        zero_trajectory = {
            "observations": np.zeros((1, self.cfg.traj_length, obs_dim)),
            "actions": np.zeros((1, self.cfg.traj_length, action_dim)),
            "rewards": np.zeros((1, self.cfg.traj_length, 1)),
            "values": np.zeros((1, self.cfg.traj_length, 1)),
        }
        history_length = self.cfg.traj_length - horizon + 1

        for k in zero_trajectory.keys():
            zero_trajectory[k][0, :history_length] = sequence_history[k][
                end_idx - history_length + 1 : end_idx + 1
            ]

        # but obs can be all traj long

        zero_trajectory["observations"][0, :smart_traj_length] = sequence_history[
            "observations"
        ][
            end_idx
            - history_length
            + 1 : end_idx
            - history_length
            + 1
            + self.cfg.traj_length
        ]

        torch_zero_trajectory = {
            (
                "states" if k == "observations" else "returns" if k == "values" else k
            ): torch.tensor(v, device=self.cfg.device, dtype=torch.float32)
            for k, v in zero_trajectory.items()
        }

        if rtg is not None:
            # Use time-related rtg for eval
            return_to_go = float(rtg)
            returns = return_to_go * np.ones((1, self.cfg.traj_length, 1))
            torch_zero_trajectory["returns"] = torch.from_numpy(returns).to(
                self.cfg.device
            )
        else:
            # Use a constant rtg for explore
            return_max = self.tokenizer_manager.tokenizers["returns"].stats.max
            return_min = self.tokenizer_manager.tokenizers["returns"].stats.min

            return_value = return_min + (return_max - return_min) * percentage
            return_to_go = float(return_value)
            returns = return_to_go * np.ones((1, self.cfg.traj_length, 1))
            torch_zero_trajectory["returns"] = torch.from_numpy(returns).to(
                self.cfg.device
            )

        full_id_mask = create_fid_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - horizon
        )

        pi_mask = create_pi_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - horizon
        )

        encode = self.tokenizer_manager.encode(torch_zero_trajectory)

        state_inference = self.tokenizer_manager.decode(self.mtm(encode, pi_mask))[
            "states"
        ]

        # full the states
        torch_zero_trajectory["states"][
            :, self.cfg.traj_length - horizon + 2 : -1, :
        ] = state_inference[:, self.cfg.traj_length - horizon + 2 : -1, :]

        torch_zero_trajectory["states"][
            :, : self.cfg.traj_length - horizon + 1, :
        ] = state_inference[:, : self.cfg.traj_length - horizon + 1, :]

        re_encode = self.tokenizer_manager.encode(torch_zero_trajectory)

        action_dist = self.tokenizer_manager.decode(self.mtm(re_encode, full_id_mask))[
            "actions"
        ]  # dist of shape(1, seq_len, act_dim)

        sample_action = action_dist.sample()[0, self.cfg.traj_length - horizon]

        eval_action = action_dist.mean[0, self.cfg.traj_length - horizon]

        if eval:
            return eval_action
        else:
            return sample_action

    @torch.no_grad()
    def action_piid_list_sample(
        self,
        sequence_history,
        percentage=1.0,
        horizon=4,
        plan=True,
        eval=False,
        rtg=None,
    ):
        if eval == True:
            assert rtg is not None

        horizon = self.cfg.horizon
        end_idx = sequence_history["path_length"]
        if end_idx + horizon < self.cfg.traj_length:
            horizon = self.cfg.traj_length - end_idx
        smart_traj_length = self.cfg.traj_length
        if end_idx + horizon > 1000:
            smart_traj_length = smart_traj_length - (end_idx + horizon - 1000)
        obs_dim = sequence_history["observations"].shape[-1]
        action_dim = sequence_history["actions"].shape[-1]
        zero_trajectory = {
            "observations": np.zeros((1, self.cfg.traj_length, obs_dim)),
            "actions": np.zeros((1, self.cfg.traj_length, action_dim)),
            "rewards": np.zeros((1, self.cfg.traj_length, 1)),
            "values": np.zeros((1, self.cfg.traj_length, 1)),
        }
        history_length = self.cfg.traj_length - horizon + 1

        for k in zero_trajectory.keys():
            zero_trajectory[k][0, :history_length] = sequence_history[k][
                end_idx - history_length + 1 : end_idx + 1
            ]

        # but obs can be all traj long

        zero_trajectory["observations"][0, :smart_traj_length] = sequence_history[
            "observations"
        ][
            end_idx
            - history_length
            + 1 : end_idx
            - history_length
            + 1
            + self.cfg.traj_length
        ]

        torch_zero_trajectory = {
            (
                "states" if k == "observations" else "returns" if k == "values" else k
            ): torch.tensor(v, device=self.cfg.device, dtype=torch.float32)
            for k, v in zero_trajectory.items()
        }

        if rtg is not None:
            # Use time-related rtg for eval
            return_to_go = float(rtg)
            returns = return_to_go * np.ones((1, self.cfg.traj_length, 1))
            torch_zero_trajectory["returns"] = torch.from_numpy(returns).to(
                self.cfg.device
            )
        else:
            # Use a constant rtg for explore
            return_max = self.tokenizer_manager.tokenizers["returns"].stats.max
            return_min = self.tokenizer_manager.tokenizers["returns"].stats.min

            return_value = return_min + (return_max - return_min) * percentage
            return_to_go = float(return_value)
            returns = return_to_go * np.ones((1, self.cfg.traj_length, 1))
            torch_zero_trajectory["returns"] = torch.from_numpy(returns).to(
                self.cfg.device
            )

        full_id_mask = create_fid_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - horizon
        )

        pi_mask = create_pi_mask(
            self.cfg.traj_length, self.cfg.device, self.cfg.traj_length - horizon
        )

        encode = self.tokenizer_manager.encode(torch_zero_trajectory)

        state_inference = self.tokenizer_manager.decode(self.mtm(encode, pi_mask))[
            "states"
        ]

        # full the states
        torch_zero_trajectory["states"][
            :, self.cfg.traj_length - horizon + 2 : -1, :
        ] = state_inference[:, self.cfg.traj_length - horizon + 2 : -1, :]

        torch_zero_trajectory["states"][
            :, : self.cfg.traj_length - horizon + 1, :
        ] = state_inference[:, : self.cfg.traj_length - horizon + 1, :]

        re_encode = self.tokenizer_manager.encode(torch_zero_trajectory)

        action_dist = self.tokenizer_manager.decode(self.mtm(re_encode, full_id_mask))[
            "actions"
        ]  # dist of shape(1, seq_len, act_dim)

        self.action_list = [
            action_dist.mean[0, self.cfg.traj_length - horizon],
            # action_dist.mean[0, self.cfg.traj_length - horizon + 1],
            # action_dist.mean[0, self.cfg.traj_length - horizon + 2],
        ]

    def compute_mtm_loss(
        self,
        batch: Dict[str, torch.Tensor],
        data_shapes,
        discrete_map,
        entropy_reg: float,
    ):
        # calculate future prediction loss
        losses = {}
        masked_losses = {}
        masked_c_losses = {}
        encoded_batch = self.tokenizer_manager.encode(batch)
        targets = encoded_batch

        masks = create_random_autoregressize_mask(
            data_shapes,
            self.cfg.mask_ratio,
            self.cfg.traj_length,
            self.cfg.device,
            self.cfg.p_weights,
        )
        preds = self.mtm(encoded_batch, masks)

        for key in targets.keys():
            target = targets[key]
            pred = preds[key]
            mask = masks[key]
            if len(mask.shape) == 1:
                # only along time dimension: repeat across the given dimension
                mask = mask[:, None].repeat(1, target.shape[2])
            elif len(mask.shape) == 2:
                pass

            batch_size, T, P, _ = target.size()
            if discrete_map[key]:
                raw_loss = nn.CrossEntropyLoss(reduction="none")(
                    pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
                ).unsqueeze(3)
            else:
                if key == "actions":
                    # sperate calc the action loss
                    raw_loss = (
                        nn.MSELoss(reduction="none")(pred.mean, target)
                        * mask[None, :, :, None]
                    )
                    losses[key] = raw_loss.mean(dim=(2, 3)).mean()

                    continue
                else:
                    # apply normalization
                    raw_loss = nn.MSELoss(reduction="none")(pred, target)
                    raw_loss = nn.MSELoss(reduction="none")(pred, target)
                    # here not taking the masked result, all the loss is calculated

            # raw_loss shape = [batch_size, T, P, 1]
            loss = raw_loss.mean(dim=(2, 3)).mean()

            masked_c_loss = (
                (raw_loss * mask[None, :, :, None]).sum(dim=(1, 2, 3)) / mask.sum()
            ).mean()
            masked_loss = (
                (raw_loss * (1 - mask[None, :, :, None])).sum(dim=(1, 2, 3))
                / (1 - mask).sum()
            ).mean()
            losses[key] = loss
            masked_c_losses[key] = masked_c_loss
            masked_losses[key] = masked_loss

            loss = torch.sum(torch.stack(list(losses.values())))

            a = targets["actions"].clip(-1 + 1e-6, 1 - 1e-6)
            a_hat_dist = preds["actions"]
            log_likelihood = a_hat_dist.log_likelihood(a)[
                :, ~masks["actions"].squeeze().to(torch.bool), :
            ].mean()
            entropy = a_hat_dist.entropy()[
                :, ~masks["actions"].squeeze().to(torch.bool), :
            ].mean()
            act_loss = -(log_likelihood + entropy_reg * entropy)
            losses["entropy"] = entropy
            losses["nll"] = -log_likelihood

            loss += act_loss

            return loss, losses, masked_losses, masked_c_losses, entropy

    def mtm_update(self, batch, data_shapes, discrete_map):
        loss, losses, masked_losses, masked_c_losses, entropy = self.compute_mtm_loss(
            batch, data_shapes, discrete_map, self.mtm.temperature().detach()
        )
        log_dict = {}
        for k, l in losses.items():
            log_dict[f"train/loss_{k}"] = l

            if k in masked_losses.keys():
                log_dict[f"train/masked_loss_{k}"] = masked_losses[k]
            if k in masked_c_losses.keys():
                log_dict[f"train/masked_c_loss_{k}"] = masked_c_losses[k]

        log_dict[f"train/loss"] = loss.item()
        log_dict["train/lr"] = self.mtm_scheduler.get_last_lr()[0]

        # backprop
        self.mtm.zero_grad(set_to_none=True)
        loss.backward()
        self.mtm_optimizer.step()
        self.mtm_scheduler.step()

        self.temp_optimizer.zero_grad()
        temperature_loss = (
            self.mtm.temperature() * (entropy - self.mtm.target_entropy).detach()
        )
        temperature_loss.backward()
        self.temp_optimizer.step()

        log_dict["train/temperature"] = self.mtm.temperature().item()
        log_dict["train/entropy"] = entropy.item()

        return log_dict

    def critic_update(self, experience: Tuple[torch.Tensor]):
        log_dict = self.iql.train(experience)

        return log_dict

    @torch.no_grad()
    def shot(
        self,
        num_episodes: int,
        episode_rtg_ref: np.ndarray,
        disable_tqdm: bool = True,
        verbose: bool = False,
        all_results: bool = False,
        num_videos: int = 3,
        way_points_path: str = None,
        two_stage: bool = False,
        list_stage: bool = False,
    ) -> Dict[str, Any]:
        log_data = {}

        for ratio in [0.9, 1.0]:
            stats: Dict[str, Any] = defaultdict(list)
            successes = None
            for i in range(num_episodes):
                current_trajectory = {
                    "observations": np.zeros(
                        (1000, self.env.observation_space.shape[0]), dtype=np.float32
                    ),
                    "actions": np.zeros(
                        (1000, self.env.action_space.shape[0]), dtype=np.float32
                    ),
                    "rewards": np.zeros((1000, 1), dtype=np.float32),
                    "values": np.zeros((1000, 1), dtype=np.float32),
                    "total_return": 0,
                    "path_length": 0,
                }

                # read 1000 obs from file
                current_trajectory["observations"] = np.loadtxt(way_points_path)

                index_jump = self.cfg.index_jump
                father_index = index_jump
                while father_index < 999:
                    for i in range(index_jump):
                        current_trajectory["observations"][
                            father_index - 1 - i
                        ] = current_trajectory["observations"][father_index]
                    father_index += index_jump + 1

                observation, done = self.env.reset(), False
                # if len(videos) < num_videos:
                #     try:
                #         imgs = [self.env.sim.render(64, 48, camera_name="track")[::-1]]
                #     except:
                #         imgs = [self.env.render()[::-1]]

                timestep = 0
                while not done and timestep < 1000:
                    current_trajectory["observations"][timestep] = observation
                    if two_stage:
                        action = self.action_piid_sample(
                            current_trajectory,
                            percentage=1.0,
                            plan=False,
                            eval=True,
                            rtg=episode_rtg_ref[timestep] * ratio,
                        )
                    elif list_stage:
                        if len(self.action_list) == 0:
                            self.action_piid_list_sample(
                                current_trajectory,
                                percentage=1.0,
                                plan=False,
                                eval=True,
                                rtg=episode_rtg_ref[timestep] * ratio,
                            )
                        action = self.action_list.pop(0)
                    else:
                        # action = self.action_id_sample(
                        #     current_trajectory,
                        #     percentage=1.0,
                        #     plan=False,
                        #     eval=True,
                        #     rtg=episode_rtg_ref[timestep] * ratio,
                        # )
                        action = self.action_sample(
                            current_trajectory,
                            percentage=1.0,
                            plan=False,
                            eval=True,
                            rtg=episode_rtg_ref[timestep] * ratio,
                        )
                    action = np.clip(action.cpu().numpy(), -1, 1)
                    new_observation, reward, done, info = self.env.step(action)
                    self.env.render(
                        mode="human", width=800, height=200, camera_id=-1
                    )  # Render the environment to visualize
                    # pause for 1 second
                    print("step: ", timestep)
                    # sleep(0.02)

                    # if timestep % 10 == 0:
                    #     plt.clf()

                    #     fig = plt.imshow(frame)
                    #     plt.show(block=True)

                    current_trajectory["actions"][timestep] = action
                    current_trajectory["rewards"][timestep] = reward
                    observation = new_observation
                    if timestep % 20 == 0:
                        print("observation: ", observation)
                    timestep += 1
                    current_trajectory["path_length"] += 1
                    if timestep > 1000:
                        break
                    # if len(videos) < num_videos:
                    #     try:
                    #         imgs.append(self.env.sim.render(64, 48, camera_name="track")[::-1])
                    #     except:
                    #         imgs.append(self.env.render()[::-1])

                # save current_trajectory["observations"] into txt
                np.savetxt(
                    "/home/hu/mtm/research/zoo/ood-traj/d4rl.txt",
                    current_trajectory["observations"],
                )
                # if len(videos) < num_videos:
                #     videos.append(np.array(imgs[:-1]))

                if "episode" in info:
                    for k in info["episode"].keys():
                        stats[k].append(float(info["episode"][k]))
                        if verbose:
                            print(f"{k}:{info['episode'][k]}")

                    ret = info["episode"]["return"]
                    mean = np.mean(stats["return"])
                    if "is_success" in info:
                        if successes is None:
                            successes = 0.0
                        successes += info["is_success"]

                else:
                    stats["return"].append(current_trajectory["rewards"].sum())
                    stats["length"].append(current_trajectory["path_length"])

            new_stats = {}
            for k, v in stats.items():
                new_stats[k + "_mean"] = float(np.mean(v))
                new_stats[k + "_std"] = float(np.std(v))

            if all_results:
                new_stats.update(stats)
            stats = new_stats
            if successes is not None:
                stats["success"] = successes / num_episodes
            for k, v in stats.items():
                log_data[f"eval_bc_{ratio}/{k}"] = v

        return log_data, stats["return_mean"]
