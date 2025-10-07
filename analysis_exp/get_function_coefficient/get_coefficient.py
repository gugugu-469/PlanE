import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from scipy.optimize import linprog
import numpy as np
from scipy.optimize import minimize
import numpy as np
from collections import defaultdict
import math
from functools import reduce
import json
def setup_lp_problem(constraints):
    """
    将不等式约束转换为线性规划问题
    """
    # 系数数量：a,b,c,d,e,f,g,h,i 共9个
    n_vars = 9
    
    # 构建不等式约束 A * x <= b
    A_ineq = []
    b_ineq = []
    
    for constraint in constraints:
        if '>' in constraint:
            parts = constraint.split('>')
            left = eval(parts[0].replace('f', '').strip())
            right = eval(parts[1].replace('f', '').strip())
            
            # f(left) > f(right) => f(left) - f(right) > 0
            # 转换为 f(left) - f(right) >= epsilon (小的正数)
            epsilon = 1e-5
            
            # 计算系数
            coeff_diff = calculate_coefficient_difference(left, right)
            A_ineq.append([-c for c in coeff_diff])  # 转换为 -coeff_diff * x <= -epsilon
            b_ineq.append(-epsilon)
            
        elif '<' in constraint:
            parts = constraint.split('<')
            left = eval(parts[0].replace('f', '').strip())
            right = eval(parts[1].replace('f', '').strip())
            
            # f(left) < f(right) => f(right) - f(left) > 0
            epsilon = 1e-5
            coeff_diff = calculate_coefficient_difference(right, left)
            A_ineq.append([-c for c in coeff_diff])
            b_ineq.append(-epsilon)
    
    # 变量边界：-10 <= x_i <= 10
    bounds = [(-10, 10) for _ in range(n_vars)]
    
    # 目标函数：最小化系数变化（这里使用零目标，因为我们只需要可行解）
    c = np.zeros(n_vars)
    
    return c, A_ineq, b_ineq, bounds

def calculate_coefficient_difference(x, y):
    """
    计算 f(x) - f(y) 的系数
    x = (x1, x2, x3), y = (y1, y2, y3)
    """
    x1, x2, x3 = x
    y1, y2, y3 = y
    
    # 二次项系数
    a_coeff = x1**2 - y1**2
    b_coeff = x2**2 - y2**2
    c_coeff = x3**2 - y3**2
    
    # 交叉项系数
    d_coeff = x1*x2 - y1*y2
    e_coeff = x1*x3 - y1*y3
    f_coeff = x2*x3 - y2*y3
    
    # 一次项系数
    g_coeff = x1 - y1
    h_coeff = x2 - y2
    i_coeff = x3 - y3
    
    return [a_coeff, b_coeff, c_coeff, d_coeff, e_coeff, f_coeff, g_coeff, h_coeff, i_coeff]

def solve_with_linprog(constraints):
    """使用线性规划求解"""
    c, A_ineq, b_ineq, bounds = setup_lp_problem(constraints)
    
    # 解决线性规划问题
    result = linprog(c, A_ub=A_ineq, b_ub=b_ineq, bounds=bounds, method='highs')
    
    if result.success:
        return result.x
    else:
        return None
import numpy as np
from scipy.optimize import minimize
import numpy as np
from collections import defaultdict
import math
from functools import reduce
import json


def enhanced_scale(arr, scale_min=1, scale_max=10):
    """
    改进的归一化函数：每个维度独立标准化，再缩放到1-10之间
    """
    arr = np.array(arr)
    scaled = np.zeros_like(arr)
    
    for i in range(arr.shape[1]):
        col = arr[:, i]
        # 标准化：均值为0，方差为1
        col_standardized = (col - np.mean(col)) / np.std(col)
        # 缩放到1-10
        scaled[:, i] = scale_min + (col_standardized - np.min(col_standardized)) * (scale_max - scale_min) / (np.max(col_standardized) - np.min(col_standardized))
    
    return np.round(scaled).astype(int)  # 取整保证整数系数

import numpy as np
import torch
import torch.optim as optim

class QuadraticModel(torch.nn.Module):
    def __init__(self, device):
        super(QuadraticModel, self).__init__()
        self.device = device
        self.coeffs = torch.nn.Parameter(torch.randn(9, device=device) * 0.1)
        # 初始化系数在[-1, 1]范围内
    
    def forward(self, x1, x2, x3):
        a, b, c, d, e, f_val, g, h, i = self.coeffs
        return (a * x1**2 + b * x2**2 + c * x3**2 + 
                d * x1*x2 + e * x1*x3 + f_val * x2*x3 + 
                g * x1 + h * x2 + i * x3)
    
    def get_coefficients(self):
        return self.coeffs.detach().cpu().numpy()

def parse_constraint(constraint):
    """解析约束条件"""
    if '>' in constraint:
        parts = constraint.split('>')
        left = eval(parts[0].replace('f', '').strip())
        right = eval(parts[1].replace('f', '').strip())
        return left, right, '>'
    elif '<' in constraint:
        parts = constraint.split('<')
        left = eval(parts[0].replace('f', '').strip())
        right = eval(parts[1].replace('f', '').strip())
        return left, right, '<'
    return None

def solve_with_gradient_descent(constraints, num_epochs=10000, lr=0.01):
    """使用梯度下降求解"""
    # 检测可用设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = QuadraticModel(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    parsed_constraints = [parse_constraint(c) for c in constraints]
    
    # 将约束转换为张量并移动到设备
    tensor_constraints = []
    for left, right, op in parsed_constraints:
        x1_l, x2_l, x3_l = left
        x1_r, x2_r, x3_r = right
        
        # 创建张量并移动到设备
        left_tensor = (torch.tensor(x1_l, dtype=torch.float32, device=device),
                      torch.tensor(x2_l, dtype=torch.float32, device=device),
                      torch.tensor(x3_l, dtype=torch.float32, device=device))
        
        right_tensor = (torch.tensor(x1_r, dtype=torch.float32, device=device),
                       torch.tensor(x2_r, dtype=torch.float32, device=device),
                       torch.tensor(x3_r, dtype=torch.float32, device=device))
        
        tensor_constraints.append((left_tensor, right_tensor, op))
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for left_tensor, right_tensor, op in tensor_constraints:
            x1_l, x2_l, x3_l = left_tensor
            x1_r, x2_r, x3_r = right_tensor
            
            left_val = model(x1_l, x2_l, x3_l)
            right_val = model(x1_r, x2_r, x3_r)
            
            if op == '>':
                # 我们希望 left_val > right_val
                # 如果不满足，则施加惩罚
                loss = torch.relu(right_val - left_val + 1e-5)
            else:  # '<'
                # 我们希望 left_val < right_val
                loss = torch.relu(left_val - right_val + 1e-5)
            
            total_loss += loss
        
        # 添加边界约束惩罚
        boundary_penalty = torch.sum(torch.relu(torch.abs(model.coeffs) - 10))
        total_loss += boundary_penalty * 0.1
        
        if total_loss.item() < 1e-8:
            print(f"在 epoch {epoch} 收敛")
            break
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(model.get_coefficients())
            print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
    
    # 检查最终系数是否在范围内
    coeffs = model.get_coefficients()
    if np.all(np.abs(coeffs) <= 10.0 + 1e-5):
        return coeffs
    else:
        print("系数超出范围")
        return None

import numpy as np
from deap import base, creator, tools, algorithms

def evaluate(individual, constraints):
    """评估个体的适应度（违反约束的数量）"""
    violations = 0
    
    for constraint in constraints:
        if '>' in constraint:
            parts = constraint.split('>')
            left = eval(parts[0].replace('f', '').strip())
            right = eval(parts[1].replace('f', '').strip())
            
            left_val = f(left[0], left[1], left[2], individual)
            right_val = f(right[0], right[1], right[2], individual)
            
            if left_val <= right_val:
                violations += 1
                
        elif '<' in constraint:
            parts = constraint.split('<')
            left = eval(parts[0].replace('f', '').strip())
            right = eval(parts[1].replace('f', '').strip())
            
            left_val = f(left[0], left[1], left[2], individual)
            right_val = f(right[0], right[1], right[2], individual)
            
            if left_val >= right_val:
                violations += 1
    
    return violations,

# common = 'CMeIE'
# need_str = 'qwen3'

common = 'qwen3'
need_str = 'CMeIE'
best_str_list = []
other_str_list = []

pred_best_str_list = []
pred_other_str_list = []

names = []
import os
for read_file in os.listdir('./f1'):
    if common in read_file:
        names.append(read_file)
        print(read_file)
        with open(os.path.join('./f1', read_file), 'r') as f:
            datas = json.load(f)
            best_str_list.append(datas['best_str_list'])
            other_str_list.append(datas['other_str_list'])
        if need_str in read_file:
            with open(os.path.join('./f1', read_file), 'r') as f:
                datas = json.load(f)
                pred_best_str_list.append(datas['best_str_list'])
                pred_other_str_list.append(datas['other_str_list'])
                
constraints = []
for best_str, other_str in zip(best_str_list, other_str_list):
    best = best_str.split('\t')
    best = [float(item) for item in best]
    others = []
    for item in other_str:
        if item == '':
            continue
        tmp_list = item.split('\t')
        tmp_list = [float(_) for _ in tmp_list]
        others.append(tmp_list)
    
    # 删除best那组数据
    others = [other for other in others if not np.allclose(other, best)]
    print(len(others))
    # 合并best和others
    all_solutions = [best] + others

    # 归一化处理所有数据
    processed_solutions = enhanced_scale(all_solutions, scale_min=1, scale_max=10)

    # 提取归一化后的best和others
    processed_best = processed_solutions[0]
    processed_others = processed_solutions[1:]

    processed_best = [str(_) for _ in processed_best]
    for other in processed_others:
        other = [str(_) for _ in other]

        constraints.append('f({}) > f({})'.format(','.join(processed_best), ','.join(other)))
    print('best:{}'.format(best))
    print('others:{}'.format(others))
    print('processed_best:{}'.format(processed_best))
    print('processed_others:{}'.format(processed_others))
# 验证函数
import numpy as np

def f(x1, x2, x3, coeffs):
    """计算二次函数值"""
    a, b, c, d, e, f_val, g, h, i = coeffs
    return (a * x1**2 + b * x2**2 + c * x3**2 + 
            d * x1*x2 + e * x1*x3 + f_val * x2*x3 + 
            g * x1 + h * x2 + i * x3)

def parse_constraint(constraint):
    """解析单个约束条件"""
    constraint = constraint.strip()
    if '>' in constraint:
        parts = constraint.split('>')
        operator = '>'
    elif '<' in constraint:
        parts = constraint.split('<')
        operator = '<'
    else:
        raise ValueError(f"无法解析约束条件: {constraint}")
    
    # 提取函数参数
    left_func = parts[0].strip()
    right_func = parts[1].strip()
    
    # 解析函数调用，例如 f(1,2,3) -> (1, 2, 3)
    def parse_function_call(func_str):
        func_str = func_str.replace('f', '').replace('(', '').replace(')', '')
        return tuple(map(float, func_str.split(',')))
    
    left_args = parse_function_call(left_func)
    right_args = parse_function_call(right_func)
    
    return left_args, right_args, operator

def verify_constraints(coeffs, constraints, tolerance=1e-10):
    """
    验证所有约束条件是否满足
    
    参数:
    coeffs: 系数列表 [a, b, c, d, e, f, g, h, i]
    constraints: 约束条件列表
    tolerance: 容忍误差，用于处理浮点数精度问题
    
    返回:
    result: 验证结果字典
    """
    results = {
        'total_constraints': len(constraints),
        'satisfied_constraints': 0,
        'violated_constraints': 0,
        'constraint_details': [],
        'all_satisfied': True
    }
    
    for i, constraint in enumerate(constraints):
        try:
            left_args, right_args, operator = parse_constraint(constraint)
            
            # 计算函数值
            left_val = f(left_args[0], left_args[1], left_args[2], coeffs)
            right_val = f(right_args[0], right_args[1], right_args[2], coeffs)
            
            # 检查约束条件
            if operator == '>':
                satisfied = (left_val - right_val) > -tolerance
                margin = left_val - right_val
            else:  # operator == '<'
                satisfied = (right_val - left_val) > -tolerance
                margin = right_val - left_val
            
            # 记录结果
            constraint_detail = {
                'index': i + 1,
                'constraint': constraint,
                'left_value': left_val,
                'right_value': right_val,
                'margin': margin,
                'satisfied': satisfied,
                'tolerance_violation': abs(margin) < tolerance if satisfied else False
            }
            
            results['constraint_details'].append(constraint_detail)
            
            if satisfied:
                results['satisfied_constraints'] += 1
            else:
                results['violated_constraints'] += 1
                results['all_satisfied'] = False
                
        except Exception as e:
            print(f"解析约束条件时出错: {constraint}")
            print(f"错误信息: {e}")
    
    return results

def print_verification_summary(results, detailed=False):
    """打印验证结果摘要"""
    print("=" * 60)
    print("约束条件验证结果")
    print("=" * 60)
    print(f"总约束数量: {results['total_constraints']}")
    print(f"满足约束数量: {results['satisfied_constraints']}")
    print(f"违反约束数量: {results['violated_constraints']}")
    print(f"所有约束是否满足: {'是' if results['all_satisfied'] else '否'}")
    print("=" * 60)
    
    if detailed and results['constraint_details']:
        print("\n详细约束验证结果:")
        print("-" * 100)
        print(f"{'序号':<4} {'约束条件':<30} {'左值':<12} {'右值':<12} {'差值':<12} {'状态':<8}")
        print("-" * 100)
        
        for detail in results['constraint_details']:
            status = "满足" if detail['satisfied'] else "违反"
            if detail['satisfied'] and detail['tolerance_violation']:
                status = "边界(满足)"
            
            print(f"{detail['index']:<4} {detail['constraint']:<30} "
                  f"{detail['left_value']:<12.6f} {detail['right_value']:<12.6f} "
                  f"{detail['margin']:<12.6f} {status:<8}")
    
    # 打印违反的约束（如果有）
    if results['violated_constraints'] > 0:
        print(f"\n违反的约束条件:")
        print("-" * 60)
        for detail in results['constraint_details']:
            if not detail['satisfied']:
                print(f"约束 {detail['index']}: {detail['constraint']}")
                print(f"  左值: {detail['left_value']:.6f}, 右值: {detail['right_value']:.6f}, "
                      f"要求差值: {abs(detail['margin']):.6f}")

def verify_coefficient_bounds(coeffs, lower_bound=-10, upper_bound=10):
    """验证系数是否在指定范围内"""
    violations = []
    coefficient_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    
    for i, (name, value) in enumerate(zip(coefficient_names, coeffs)):
        if value < lower_bound or value > upper_bound:
            violations.append({
                'coefficient': name,
                'value': value,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
    
    return violations




lrs = [
    1e-3,
    1e-4,
    1e-5,
]


for lr in lrs:
    print('LR:{}'.format(lr))
    if not constraints:
        print("请添加约束条件")
    else:
        # 尝试线性规划方法
        print("尝试线性规划方法...")
        coefficients_1 = solve_with_linprog(constraints)
        
        if coefficients_1 is None:
            print("线性规划无解，尝试梯度下降方法...")
            coefficients_1 = solve_with_gradient_descent(constraints, num_epochs = 50000, lr = lr)
        print('结果:{}'.format(coefficients_1))
        if coefficients_1 is not None:
            a, b, c, d, e, f_val, g, h, i = coefficients_1
            print("找到满足条件的系数:")
            print(f"a = {a:.6f}")
            print(f"b = {b:.6f}")
            print(f"c = {c:.6f}")
            print(f"d = {d:.6f}")
            print(f"e = {e:.6f}")
            print(f"f = {f_val:.6f}")
            print(f"g = {g:.6f}")
            print(f"h = {h:.6f}")
            print(f"i = {i:.6f}")
        else:
            print("未找到满足所有约束的解")

    
    coefficients_1
    constraints
    example_coeffs = coefficients_1
        
    # 示例约束条件（请替换为您的实际约束）
    example_constraints = constraints

    # 验证约束条件
    verification_results = verify_constraints(example_coeffs, example_constraints)

    # 打印摘要
    print_verification_summary(verification_results, detailed=True)

    # 验证系数范围
    bound_violations = verify_coefficient_bounds(example_coeffs)
    if bound_violations:
        print(f"\n系数范围违反:")
        for violation in bound_violations:
            print(f"{violation['coefficient']} = {violation['value']:.6f} "
                    f"(范围: [{violation['lower_bound']}, {violation['upper_bound']}])")
    else:
        print(f"\n所有系数都在指定范围内")

    # 综合验证结果
    print(f"\n综合验证结果: {'通过' if verification_results['all_satisfied'] and not bound_violations else '未通过'}")




