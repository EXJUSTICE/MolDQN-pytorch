import torch
import torch.nn as nn
import numpy as np
import torch.optim as opt
import utils
import hyp
from dqn import MolDQN
from rdkit import Chem
from rdkit.Chem import QED
from environment import Molecule
from baselines.deepq import replay_buffer
from molecules import penalized_logp

from rdkit import DataStructs
from rdkit.Chem import AllChem

REPLAY_BUFFER_CAPACITY = hyp.replay_buffer_size

"""
Single Objective Optimization Molecule Class
"""
class QEDRewardMolecule(Molecule):
    """The molecule whose reward is the QED."""

    def __init__(self, discount_factor,reward_type, **kwargs):
        """Initializes the class.

    Args:
      discount_factor: Float. The discount factor. We only
        care about the molecule at the end of modification.
        In order to prevent a myopic decision, we discount
        the reward at each step by a factor of
        discount_factor ** num_steps_left,
        this encourages exploration with emphasis on long term rewards.
      reward_type: String.  Argument to be passed to set the
        environment reward. Either QED or Penalized LogP.

      **kwargs: The keyword arguments passed to the base class.
    """
        super(QEDRewardMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor
        self.reward_type = reward_type

    def _reward(self):
        """Reward of a state. 
        Two rewards implemented
        Input: "QED"
        Returns:
        Float. QED of the current state.

        Input: "LogP"
        Returns: 
        Float. The normalized penalized logP value.
        """
        if self.reward_type == "QED":
          molecule = Chem.MolFromSmiles(self._state)
          if molecule is None:
              return 0.0
          qed = QED.qed(molecule)
          return qed * self.discount_factor ** (self.max_steps - self.num_steps_taken)
        if self.reward_type =="LogP":
          molecule = Chem.MolFromSmiles(self._state)
          if molecule is None:
            return 0.0
          return penalized_logp(molecule)
 
"""
Multi Objective Optimization Molecule Class
"""
class MultiObjectiveRewardMolecule(Molecule):
  """Defines the subclass of generating a molecule with a specific reward.
  The reward is defined as a scalar
    reward = weight * similarity_score + (1 - weight) *  qed_score
  """

  def __init__(self, target_molecule, similarity_weight, discount_factor,
               **kwargs):
    """Initializes the class.
    Args:
      target_molecule: SMILES string. The target molecule against which we
        calculate the similarity.
      similarity_weight: Float. The weight applied similarity_score.
      discount_factor: Float. The discount factor applied on reward.
      **kwargs: The keyword arguments passed to the parent class.
    """
    super(MultiObjectiveRewardMolecule, self).__init__(**kwargs)
    #Replaced chemMolFromSmiles Argument from TargetMolecule to 'c1ccccc1'
    target_molecule = Chem.MolFromSmiles('c1ccccc1')
    self._target_mol_fingerprint = self.get_fingerprint(target_molecule)
    self._sim_weight = similarity_weight
    self._discount_factor = discount_factor

  def get_fingerprint(self, molecule):
    """Gets the morgan fingerprint of the target molecule.
    Args:
      molecule: Chem.Mol. The current molecule.
    Returns:
      rdkit.ExplicitBitVect. The fingerprint of the target.
    """
    return AllChem.GetMorganFingerprint(molecule, radius=2)

  def get_similarity(self, smiles):
    """Gets the similarity between the current molecule and the target molecule.
    Args:
      smiles: String. The SMILES string for the current molecule.
    Returns:
      Float. The Tanimoto similarity.
    """

    structure = Chem.MolFromSmiles(smiles)
    if structure is None:
      return 0.0
    fingerprint_structure = self.get_fingerprint(structure)

    return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint,
                                          fingerprint_structure)

  def _reward(self):
    """Calculates the reward of the current state.
    The reward is defined as a tuple of the similarity and QED value.
    Returns:
      A tuple of the similarity and qed value
    """
    # calculate similarity.
    # if the current molecule does not contain the scaffold of the target,
    # similarity is zero.
    if self._state is None:
      return 0.0
    mol = Chem.MolFromSmiles(self._state)
    if mol is None:
      return 0.0
    similarity_score = self.get_similarity(self._state)
    # calculate QED
    qed_value = QED.qed(mol)
    reward = (
        similarity_score * self._sim_weight +
        qed_value * (1 - self._sim_weight))
    discount = self._discount_factor**(self.max_steps - self._counter)
    return reward * discount



class Agent(object):
    def __init__(self, input_length, output_length, device):
        self.device = device
        self.dqn, self.target_dqn = (
            MolDQN(input_length, output_length).to(self.device),
            MolDQN(input_length, output_length).to(self.device),
        )
        for p in self.target_dqn.parameters():
            p.requires_grad = False
        self.replay_buffer = replay_buffer.ReplayBuffer(REPLAY_BUFFER_CAPACITY)
        self.optimizer = getattr(opt, hyp.optimizer)(
            self.dqn.parameters(), lr=hyp.learning_rate
        )

    def get_action(self, observations, epsilon_threshold):

        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, observations.shape[0])
        else:
            q_value = self.dqn.forward(observations.to(self.device)).cpu()
            action = torch.argmax(q_value).detach().numpy()

        return action

    def update_params(self, batch_size, gamma, polyak):
        # update target network

        # sample batch of transitions
        states, _, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        q_t = torch.zeros(batch_size, 1, requires_grad=False)
        v_tp1 = torch.zeros(batch_size, 1, requires_grad=False)
        for i in range(batch_size):
            state = (
                torch.FloatTensor(states[i])
                .reshape(-1, hyp.fingerprint_length + 1)
                .to(self.device)
            )
            q_t[i] = self.dqn(state)

            next_state = (
                torch.FloatTensor(next_states[i])
                .reshape(-1, hyp.fingerprint_length + 1)
                .to(self.device)
            )
            v_tp1[i] = torch.max(self.target_dqn(next_state))

        rewards = torch.FloatTensor(rewards).reshape(q_t.shape).to(self.device)
        q_t = q_t.to(self.device)
        v_tp1 = v_tp1.to(self.device)
        dones = torch.FloatTensor(dones).reshape(q_t.shape).to(self.device)

        # # get q values
        q_tp1_masked = (1 - dones) * v_tp1
        q_t_target = rewards + gamma * q_tp1_masked
        td_error = q_t - q_t_target

        q_loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error * td_error,
            1.0 * (torch.abs(td_error) - 0.5),
        )
        q_loss = q_loss.mean()

        # backpropagate
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        return q_loss
