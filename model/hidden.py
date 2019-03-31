import numpy as np

import torch
import torch.nn as nn

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder


class HiDDen(object):
    def __init__(self, config: HiDDenConfiguration, device: torch.device):
        self.enc_dec = EncoderDecoder(config).to(device)
        self.discr = Discriminator(config).to(device)
        self.opt_enc_dec = torch.optim.Adam(self.enc_dec.parameters())
        self.opt_discr = torch.optim.Adam(self.discr.parameters())

        self.config = config
        self.device = device
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        self.cover_label = 1
        self.encod_label = 0

    def train_on_batch(self, batch: list):
        '''
        Trains the network on a single batch consistring images and messages
        '''
        images, messages = batch
        batch_size = images.shape[0]
        self.enc_dec.train()
        self.discr.train()

        with torch.enable_grad():
            # ---------- Train the discriminator----------
            self.opt_discr.zero_grad()

            # train on cover
            d_target_label_cover = torch.full((batch_size, 1),
                                              self.cover_label,
                                              device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1),
                                                self.encod_label,
                                                device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1),
                                                self.cover_label,
                                                device=self.device)

            d_on_cover = self.discr(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover,
                                                        d_target_label_cover)
            d_loss_on_cover.backward()

            # train on fake
            encoded_images, decoded_messages = self.enc_dec(images, messages)
            d_on_encoded = self.discr(encoded_images.detach())
            d_loss_on_encod = self.bce_with_logits_loss(d_on_encoded,
                                                        d_target_label_encoded)
            d_loss_on_encod.backward()
            self.opt_discr.step()

            #---------- Train the generator----------
            self.opt_enc_dec.zero_grad()

            d_on_encoded_for_enc = self.discr(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc,
                                                   g_target_label_encoded)
            g_loss_enc = self.mse_loss(encoded_images, images)
            g_loss_dec = self.mse_loss(decoded_messages, messages)

            g_loss = self.config.adversarial_loss * g_loss_adv \
                    + self.config.encoder_loss * g_loss_enc \
                    + self.config.decoder_loss * g_loss_dec
            g_loss.backward()
            self.opt_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) \
                      / (batch_size * messages.shape[1])

        losses = {
            'loss': g_loss.item(),
            'encoder_mse': g_loss_enc.item(),
            'decoder_mse': g_loss_dec.item(),
            'bitwise-error': bitwise_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encod.item()
        }

        return losses, (encoded_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        '''Run validation on a batch consist of [images, messages]'''
        images, messages = batch
        batch_size = images.shape[0]

        self.enc_dec.eval()
        self.discr.eval()

        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1),
                                              self.cover_label,
                                              device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1),
                                                self.encod_label,
                                                device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1),
                                                self.cover_label,
                                                device=self.device)

            d_on_cover = self.discr(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover,
                                                        d_target_label_cover)

            encoded_images, decoded_messages = self.enc_dec(images, messages)
            d_on_encoded = self.discr(encoded_images)
            d_loss_on_encod = self.bce_with_logits_loss(d_on_encoded,
                                                        d_target_label_encoded)

            d_on_encoded_for_enc = self.discr(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc,
                                                   g_target_label_encoded)
            g_loss_enc = self.mse_loss(encoded_images, images)
            g_loss_dec = self.mse_loss(decoded_messages, messages)

            g_loss = self.config.adversarial_loss * g_loss_adv \
                    + self.config.encoder_loss * g_loss_enc \
                    + self.config.decoder_loss * g_loss_dec

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy()))\
                     / (batch_size * messages.shape[1])

        losses = {
            'loss': g_loss.item(),
            'encoder_mse': g_loss_enc.item(),
            'decoder_mse': g_loss_dec.item(),
            'bitwise-err': bitwise_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_enced_bce': d_loss_on_encod.item()
        }

        return losses, (encoded_images, decoded_messages)

    def to_stirng(self):
        return f'{str(self.enc_dec)}\n{str(self.discr)}'
