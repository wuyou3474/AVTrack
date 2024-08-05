import torch
import torch.nn as nn
import torch.nn.functional as F






class DJSLoss(nn.Module):
    """Jensen Shannon Divergence loss"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, T: torch.Tensor, T_prime: torch.Tensor) -> float:
        """Estimator of the Jensen Shannon Divergence see paper equation (2)

        Args:
            T (torch.Tensor): Statistique network estimation from the marginal distribution P(x)P(z)
            T_prime (torch.Tensor): Statistique network estimation from the joint distribution P(xz)

        Returns:
            float: DJS estimation value
        """
        joint_expectation = (-F.softplus(-T)).mean()
        marginal_expectation = F.softplus(T_prime).mean()
        mutual_info = joint_expectation - marginal_expectation

        return -mutual_info


class DiscriminatorLoss(nn.Module):
    """Basic discriminator GAN loss """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> float:
        """Discriminator loss gan

        Args:
            real_logits (torch.Tensor): Sample from the real distribution here from P(Sx)P(Ex)
            fake_logits (torch.Tensor): Sample from the fake (generated) distribution here from P(SxEx)

        Returns:
            float: Discriminator loss value
        """

        # Discriminator should predict real logits as logits from the real distribution
        discriminator_real = F.binary_cross_entropy_with_logits(
            input=real_logits, target=torch.ones_like(real_logits)
        )
        # Discriminator should predict fake logits as logits from the generated distribution
        discriminator_fake = F.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.zeros_like(fake_logits)
        )
        discriminator_loss = discriminator_real.mean() + discriminator_fake.mean()

        return discriminator_loss


class GeneratorLoss(nn.Module):
    """Basic generator GAN loss """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, fake_logits: torch.Tensor) -> float:
        """Generator loss

        Args:
            fake_logits (torch.Tensor): Sample from the fake (generated) distribution here from P(SxEx)

        Returns:
            float: Generator loss value
        """
        # Discriminator should generate fake logits that fool the discriminator
        generator_loss = F.binary_cross_entropy_with_logits(
            input=fake_logits, target=torch.ones_like(fake_logits)
        )
        return generator_loss
