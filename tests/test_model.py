import pytest
import torch
from torch_geometric.data import Data, Batch

from src.model import PatchEncoder, SimVQCodebook, PatchDecoder, MeshLexVQVAE


# ── Encoder Tests ────────────────────────────────────────────────────────────

def test_encoder_output_shape():
    """Encoder should produce (B, embed_dim) from batched graph."""
    batch_size = 4
    graphs = []
    for _ in range(batch_size):
        n_faces = 30
        x = torch.randn(n_faces, 15)
        src = torch.randint(0, n_faces, (n_faces * 2,))
        dst = torch.randint(0, n_faces, (n_faces * 2,))
        edge_index = torch.stack([src, dst])
        graphs.append(Data(x=x, edge_index=edge_index))

    batch = Batch.from_data_list(graphs)
    encoder = PatchEncoder(in_dim=15, hidden_dim=256, out_dim=128)
    out = encoder(batch.x, batch.edge_index, batch.batch)

    assert out.shape == (batch_size, 128)


def test_encoder_deterministic():
    """Same input should give same output."""
    torch.manual_seed(42)
    x = torch.randn(20, 15)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data])

    encoder = PatchEncoder()
    encoder.eval()
    with torch.no_grad():
        out1 = encoder(batch.x, batch.edge_index, batch.batch)
        out2 = encoder(batch.x, batch.edge_index, batch.batch)
    assert torch.allclose(out1, out2)


# ── Codebook Tests ───────────────────────────────────────────────────────────

def test_codebook_output_shape():
    codebook = SimVQCodebook(K=64, dim=128)
    z = torch.randn(8, 128)
    quantized, indices = codebook(z)
    assert quantized.shape == (8, 128)
    assert indices.shape == (8,)
    assert indices.min() >= 0
    assert indices.max() < 64


def test_codebook_straight_through_gradient():
    """Gradients should flow through quantization via straight-through."""
    codebook = SimVQCodebook(K=64, dim=128)
    z = torch.randn(4, 128, requires_grad=True)
    quantized, _ = codebook(z)
    loss = quantized.sum()
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_codebook_utilization():
    """With diverse inputs, utilization should be non-trivial."""
    codebook = SimVQCodebook(K=32, dim=16)
    z = torch.randn(256, 16)
    _, indices = codebook(z)
    unique_codes = indices.unique().numel()
    assert unique_codes >= 8, f"Only {unique_codes}/32 codes used"


# ── Decoder Tests ────────────────────────────────────────────────────────────

def test_decoder_output_shape():
    decoder = PatchDecoder(embed_dim=128, max_vertices=60)
    z = torch.randn(4, 128)
    n_vertices = torch.tensor([20, 30, 25, 40])
    out = decoder(z, n_vertices)
    assert out.shape == (4, 60, 3)


def test_decoder_masked_output():
    """Vertices beyond n_vertices should be zero (masked)."""
    decoder = PatchDecoder(embed_dim=128, max_vertices=60)
    z = torch.randn(2, 128)
    n_vertices = torch.tensor([10, 15])
    out = decoder(z, n_vertices)
    assert torch.allclose(out[0, 10:], torch.zeros(50, 3))
    assert torch.allclose(out[1, 15:], torch.zeros(45, 3))


# ── VQ-VAE End-to-End Tests ─────────────────────────────────────────────────

def test_vqvae_forward():
    """Full forward pass: graph input → reconstructed vertices + losses."""
    model = MeshLexVQVAE(codebook_size=64, embed_dim=128)
    graphs = []
    n_verts_list = []
    for _ in range(4):
        nf = 30
        nv = 20
        x = torch.randn(nf, 15)
        ei = torch.stack([torch.randint(0, nf, (60,)), torch.randint(0, nf, (60,))])
        graphs.append(Data(x=x, edge_index=ei))
        n_verts_list.append(nv)

    batch = Batch.from_data_list(graphs)
    n_vertices = torch.tensor(n_verts_list)
    gt_vertices = torch.randn(4, 60, 3)

    result = model(batch.x, batch.edge_index, batch.batch, n_vertices, gt_vertices)

    assert "recon_vertices" in result
    assert "total_loss" in result
    assert "recon_loss" in result
    assert "commit_loss" in result
    assert "indices" in result
    assert result["recon_vertices"].shape == (4, 60, 3)
    assert result["total_loss"].requires_grad
