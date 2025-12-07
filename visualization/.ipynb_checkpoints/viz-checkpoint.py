from graphviz import Digraph

# Create a directed graph - LEFT TO RIGHT layout (horizontal)
dot = Digraph('MSNet1D', format='png')
dot.attr(rankdir='LR', size='20,10', dpi='150')  # Left-to-right layout
dot.attr('node', shape='box', style='filled,rounded', fillcolor='white', fontsize='10', fontname='Arial')
dot.attr('edge', fontsize='9')

# ============================================================================
# INPUT
# ============================================================================
dot.node('Input', 'Input\n(111 features)', fillcolor='#E8F5E9', shape='ellipse')

# ============================================================================
# INPUT PROJECTION
# ============================================================================
dot.node('InputProj', 'Input Projection\nLinear(111→128)\nBN + ReLU + Drop', fillcolor='#F5F5F5')
dot.edge('Input', 'InputProj')

# ============================================================================
# MSBlock1 (128 → 64)
# ============================================================================
with dot.subgraph(name='cluster_MSBlock1') as ms1:
    ms1.attr(label='MSBlock1 (128 → 64)', style='rounded,filled', fillcolor='#E3F2FD', fontsize='11', fontname='Arial Bold')
    
    # Path 1: Direct (1 layer) - in_features → out_features
    ms1.node('MS1_P1', 'Path1 (Direct)\nLinear(128→64)\nBN + ReLU + Drop', fillcolor='#BBDEFB')
    
    # Path 2: Medium (2 layers) - in_features → in_features → out_features
    ms1.node('MS1_P2', 'Path2 (Medium)\nLinear(128→128)\nLinear(128→64)\nBN + ReLU + Drop', fillcolor='#BBDEFB')
    
    # Path 3: Deep with bottleneck - in_features → bottleneck → bottleneck → out_features
    # bottleneck = in_features // 2 = 128 // 2 = 64
    ms1.node('MS1_P3', 'Path3 (Bottleneck)\nLinear(128→64)\nLinear(64→64)\nLinear(64→64)\nBN + ReLU + Drop', fillcolor='#BBDEFB')
    
    # Concatenate (64*3 = 192)
    ms1.node('MS1_Cat', 'Concat\n(192)', shape='ellipse', fillcolor='#90CAF9')
    
    # Combine: Linear(192 → 64)
    ms1.node('MS1_Comb', 'Combine\nLinear(192→64)\nBN + ReLU', fillcolor='#64B5F6')
    
    ms1.edge('MS1_P1', 'MS1_Cat')
    ms1.edge('MS1_P2', 'MS1_Cat')
    ms1.edge('MS1_P3', 'MS1_Cat')
    ms1.edge('MS1_Cat', 'MS1_Comb')

# Input to all paths in MSBlock1
dot.edge('InputProj', 'MS1_P1')
dot.edge('InputProj', 'MS1_P2')
dot.edge('InputProj', 'MS1_P3')

# Skip connection for MSBlock1: Linear(128→64) since in != out
dot.node('MS1_Skip', 'Skip\nLinear(128→64)', fillcolor='#E1F5FE', shape='box')
dot.edge('InputProj', 'MS1_Skip', style='dashed', color='#1976D2')

# Add node for residual
dot.node('MS1_Add', '+', shape='circle', width='0.4', fillcolor='#FFECB3', fontsize='14')
dot.edge('MS1_Comb', 'MS1_Add')
dot.edge('MS1_Skip', 'MS1_Add', style='dashed', color='#1976D2')

# ============================================================================
# MSBlock2 (64 → 64)
# ============================================================================
with dot.subgraph(name='cluster_MSBlock2') as ms2:
    ms2.attr(label='MSBlock2 (64 → 64)', style='rounded,filled', fillcolor='#E8F5E9', fontsize='11', fontname='Arial Bold')
    
    # Path 1: Direct - 64 → 64
    ms2.node('MS2_P1', 'Path1 (Direct)\nLinear(64→64)\nBN + ReLU + Drop', fillcolor='#C8E6C9')
    
    # Path 2: Medium - 64 → 64 → 64
    ms2.node('MS2_P2', 'Path2 (Medium)\nLinear(64→64)\nLinear(64→64)\nBN + ReLU + Drop', fillcolor='#C8E6C9')
    
    # Path 3: Bottleneck - 64 → 32 → 32 → 64
    # bottleneck = 64 // 2 = 32
    ms2.node('MS2_P3', 'Path3 (Bottleneck)\nLinear(64→32)\nLinear(32→32)\nLinear(32→64)\nBN + ReLU + Drop', fillcolor='#C8E6C9')
    
    # Concatenate (64*3 = 192)
    ms2.node('MS2_Cat', 'Concat\n(192)', shape='ellipse', fillcolor='#A5D6A7')
    
    # Combine
    ms2.node('MS2_Comb', 'Combine\nLinear(192→64)\nBN + ReLU', fillcolor='#81C784')
    
    ms2.edge('MS2_P1', 'MS2_Cat')
    ms2.edge('MS2_P2', 'MS2_Cat')
    ms2.edge('MS2_P3', 'MS2_Cat')
    ms2.edge('MS2_Cat', 'MS2_Comb')

# Connect MSBlock1 output to MSBlock2 inputs
dot.edge('MS1_Add', 'MS2_P1')
dot.edge('MS1_Add', 'MS2_P2')
dot.edge('MS1_Add', 'MS2_P3')

# Skip connection for MSBlock2: Identity since in == out (64 == 64)
dot.node('MS2_Skip', 'Skip\nIdentity', fillcolor='#E8F5E9', shape='box')
dot.edge('MS1_Add', 'MS2_Skip', style='dashed', color='#388E3C')

# Add node for residual in MSBlock2
dot.node('MS2_Add', '+', shape='circle', width='0.4', fillcolor='#FFECB3', fontsize='14')
dot.edge('MS2_Comb', 'MS2_Add')
dot.edge('MS2_Skip', 'MS2_Add', style='dashed', color='#388E3C')

# ============================================================================
# ASPP Module (64 → 64)
# hidden = 64 // 4 = 16
# ============================================================================
with dot.subgraph(name='cluster_ASPP') as aspp:
    aspp.attr(label='ASPP Module (64 → 64)', style='rounded,filled', fillcolor='#FFF3E0', fontsize='11', fontname='Arial Bold')
    
    # Branch 1: 1 layer (64 → 16)
    aspp.node('ASPP_B1', 'Branch1\nLinear(64→16)\nBN + ReLU', fillcolor='#FFE0B2')
    
    # Branch 2: 2 layers (64 → 16 → 16)
    aspp.node('ASPP_B2', 'Branch2\nLinear(64→16)\nLinear(16→16)\nBN + ReLU', fillcolor='#FFE0B2')
    
    # Branch 3: 3 layers (64 → 16 → 16 → 16)
    aspp.node('ASPP_B3', 'Branch3\nLinear(64→16)\nLinear(16→16)\nLinear(16→16)\nBN + ReLU', fillcolor='#FFE0B2')
    
    # Branch 4: 1 layer (64 → 16)
    aspp.node('ASPP_B4', 'Branch4\nLinear(64→16)\nBN + ReLU', fillcolor='#FFE0B2')
    
    # Concatenate (16*4 = 64)
    aspp.node('ASPP_Cat', 'Concat\n(64)', shape='ellipse', fillcolor='#FFCC80')
    
    # Combine: Linear(64 → 64)
    aspp.node('ASPP_Comb', 'Combine\nLinear(64→64)\nBN + ReLU + Drop', fillcolor='#FFB74D')
    
    aspp.edge('ASPP_B1', 'ASPP_Cat')
    aspp.edge('ASPP_B2', 'ASPP_Cat')
    aspp.edge('ASPP_B3', 'ASPP_Cat')
    aspp.edge('ASPP_B4', 'ASPP_Cat')
    aspp.edge('ASPP_Cat', 'ASPP_Comb')

# Connect MSBlock2 output to ASPP inputs
dot.edge('MS2_Add', 'ASPP_B1')
dot.edge('MS2_Add', 'ASPP_B2')
dot.edge('MS2_Add', 'ASPP_B3')
dot.edge('MS2_Add', 'ASPP_B4')

# ============================================================================
# SHARED EMBEDDING
# ============================================================================
dot.node('Shared', 'Shared Embedding\nLinear(64→32)\nBN + ReLU + Dropout', fillcolor='#E1BEE7')
dot.edge('ASPP_Comb', 'Shared')

# ============================================================================
# DUAL OUTPUT HEADS
# ============================================================================

# Regression head
with dot.subgraph(name='cluster_reg') as reg:
    reg.attr(label='Regression Head', style='rounded,filled', fillcolor='#FFF9C4', fontsize='10')
    reg.node('Reg_L1', 'Linear(32→16)\nReLU', fillcolor='#FFF59D')
    reg.node('Reg_L2', 'Linear(16→1)', fillcolor='#FFF59D')
    reg.edge('Reg_L1', 'Reg_L2')

dot.node('Reg_Out', 'LOS\n(hours)', fillcolor='#FFEB3B', shape='ellipse')
dot.edge('Shared', 'Reg_L1')
dot.edge('Reg_L2', 'Reg_Out')

# Classification head
with dot.subgraph(name='cluster_cls') as cls:
    cls.attr(label='Classification Head', style='rounded,filled', fillcolor='#FFCDD2', fontsize='10')
    cls.node('Cls_L1', 'Linear(32→16)\nReLU', fillcolor='#EF9A9A')
    cls.node('Cls_L2', 'Linear(16→4)', fillcolor='#EF9A9A')
    cls.edge('Cls_L1', 'Cls_L2')

dot.node('Cls_Out', 'LOS Class\n(4 classes)', fillcolor='#F44336', fontcolor='white', shape='ellipse')
dot.edge('Shared', 'Cls_L1')
dot.edge('Cls_L2', 'Cls_Out')

# ============================================================================
# RENDER
# ============================================================================
dot.render('msnet1d_architecture_horizontal', view=False, cleanup=True)
print("✓ Saved: msnet1d_architecture_horizontal.png")