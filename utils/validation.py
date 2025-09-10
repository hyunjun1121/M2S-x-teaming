"""
Validation utilities for M2S-x-teaming
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any


def validate_api_keys(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    API 키 존재 여부를 검증
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        각 provider별 API 키 존재 여부
    """
    api_key_status = {}
    
    # 설정에서 사용되는 provider들 추출
    providers_to_check = set()
    
    # 각 섹션에서 provider 찾기
    for section_name, section_config in config.items():
        if isinstance(section_config, dict) and "provider" in section_config:
            provider = section_config["provider"]
            if provider in ["openai", "anthropic", "openrouter", "together"]:
                providers_to_check.add(provider)
    
    # 각 provider의 API 키 확인
    for provider in providers_to_check:
        env_var_name = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_var_name)
        api_key_status[provider] = bool(api_key and api_key.strip())
        
        if not api_key_status[provider]:
            logging.warning(f"API key not found for {provider}. Set {env_var_name} environment variable.")
    
    return api_key_status


def validate_config_structure(config: Dict[str, Any]) -> List[str]:
    """
    설정 파일 구조 검증
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        발견된 문제들의 리스트
    """
    issues = []
    
    # 필수 섹션들
    required_sections = {
        "target": ["provider", "model"],
        "evaluation": ["use_gpt_judge", "judge_model"],
        "experiment": ["base_templates", "turn_variations"],
        "multithreading": ["max_workers"]
    }
    
    # 선택적 섹션들
    optional_sections = {
        "m2s_template_generator": ["provider", "model", "temperature"],
        "logging": ["level"]
    }
    
    # 필수 섹션 검증
    for section_name, required_fields in required_sections.items():
        if section_name not in config:
            issues.append(f"Required section missing: {section_name}")
            continue
        
        section = config[section_name]
        if not isinstance(section, dict):
            issues.append(f"Section {section_name} must be a dictionary")
            continue
        
        for field in required_fields:
            if field not in section:
                issues.append(f"Required field missing: {section_name}.{field}")
    
    # 데이터 타입 검증
    if "multithreading" in config:
        max_workers = config["multithreading"].get("max_workers")
        if max_workers is not None:
            if not isinstance(max_workers, int) or max_workers < 1:
                issues.append("multithreading.max_workers must be a positive integer")
    
    if "experiment" in config:
        base_templates = config["experiment"].get("base_templates")
        if base_templates is not None:
            if not isinstance(base_templates, list):
                issues.append("experiment.base_templates must be a list")
            elif not all(isinstance(t, str) for t in base_templates):
                issues.append("experiment.base_templates must contain only strings")
    
    return issues


def validate_file_paths(config: Dict[str, Any]) -> List[str]:
    """
    설정에 포함된 파일 경로들 검증
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        발견된 문제들의 리스트
    """
    issues = []
    
    # 검증할 파일 경로들
    file_path_fields = [
        ("experiment", "multiturn_dataset_path"),
        ("experiment", "results_dir"),
        ("experiment", "templates_dir"),
        ("logging", "log_dir")
    ]
    
    for section_name, field_name in file_path_fields:
        if section_name in config and field_name in config[section_name]:
            file_path = config[section_name][field_name]
            
            if not isinstance(file_path, str):
                issues.append(f"{section_name}.{field_name} must be a string")
                continue
            
            # 결과 디렉토리는 자동 생성되므로 존재 검사하지 않음
            if field_name in ["results_dir", "templates_dir", "log_dir"]:
                try:
                    # 디렉토리 생성 가능한지만 확인
                    parent_dir = os.path.dirname(file_path)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory for {section_name}.{field_name}: {e}")
            
            # 데이터셋 파일은 존재해야 함
            elif field_name == "multiturn_dataset_path":
                if not os.path.exists(file_path):
                    issues.append(f"Dataset file not found: {file_path}")
    
    return issues


def validate_templates_file(templates_file: str) -> List[str]:
    """
    템플릿 파일 구조 검증
    
    Args:
        templates_file: 템플릿 파일 경로
        
    Returns:
        발견된 문제들의 리스트
    """
    issues = []
    
    if not os.path.exists(templates_file):
        issues.append(f"Templates file not found: {templates_file}")
        return issues
    
    try:
        with open(templates_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        issues.append(f"Invalid JSON in templates file: {e}")
        return issues
    except Exception as e:
        issues.append(f"Failed to read templates file: {e}")
        return issues
    
    if not isinstance(data, dict):
        issues.append("Templates file must contain a JSON object")
        return issues
    
    if "templates" not in data:
        issues.append("Templates file missing 'templates' field")
        return issues
    
    templates = data["templates"]
    if not isinstance(templates, list):
        issues.append("'templates' field must be a list")
        return issues
    
    # 각 템플릿 검증
    required_template_fields = ["name", "description", "template"]
    for i, template in enumerate(templates):
        if not isinstance(template, dict):
            issues.append(f"Template {i} is not a dictionary")
            continue
        
        for field in required_template_fields:
            if field not in template:
                issues.append(f"Template {i} missing required field: {field}")
        
        # 이름 중복 검사
        template_names = [t.get("name") for t in templates if isinstance(t, dict) and "name" in t]
        if len(template_names) != len(set(template_names)):
            issues.append("Duplicate template names found")
    
    return issues


def check_system_requirements() -> Dict[str, Any]:
    """
    시스템 요구사항 검사
    
    Returns:
        시스템 상태 정보
    """
    status = {
        "python_version": sys.version,
        "python_version_ok": sys.version_info >= (3, 8),
        "required_modules": {},
        "optional_modules": {},
        "disk_space_ok": True,
        "memory_ok": True
    }
    
    # 필수 모듈 검사
    required_modules = ["json", "yaml", "logging", "concurrent.futures", "threading"]
    for module_name in required_modules:
        try:
            __import__(module_name)
            status["required_modules"][module_name] = True
        except ImportError:
            status["required_modules"][module_name] = False
    
    # 선택적 모듈 검사
    optional_modules = ["matplotlib", "seaborn", "pandas", "numpy"]
    for module_name in optional_modules:
        try:
            __import__(module_name)
            status["optional_modules"][module_name] = True
        except ImportError:
            status["optional_modules"][module_name] = False
    
    # 디스크 공간 검사 (간단한 버전)
    try:
        import shutil
        free_space = shutil.disk_usage('.').free
        status["free_disk_space_gb"] = free_space / (1024**3)
        status["disk_space_ok"] = free_space > 1024**3  # 1GB 이상
    except Exception:
        status["free_disk_space_gb"] = None
    
    return status


def validate_environment() -> bool:
    """
    전체 환경 검증
    
    Returns:
        검증 통과 여부
    """
    all_good = True
    
    print("=== M2S-x-teaming Environment Validation ===")
    
    # 시스템 요구사항 검사
    sys_status = check_system_requirements()
    
    print(f"\nPython Version: {sys_status['python_version']}")
    if not sys_status["python_version_ok"]:
        print(" Python 3.8 or higher is required")
        all_good = False
    else:
        print(" Python version OK")
    
    # 필수 모듈 검사
    print("\nRequired Modules:")
    for module, available in sys_status["required_modules"].items():
        status = "" if available else ""
        print(f"  {status} {module}")
        if not available:
            all_good = False
    
    # 선택적 모듈 검사
    print("\nOptional Modules:")
    for module, available in sys_status["optional_modules"].items():
        status = "" if available else "⚠️ "
        print(f"  {status} {module}")
    
    # 디스크 공간
    if sys_status["free_disk_space_gb"]:
        print(f"\nDisk Space: {sys_status['free_disk_space_gb']:.1f} GB available")
        if sys_status["disk_space_ok"]:
            print(" Sufficient disk space")
        else:
            print("⚠️  Low disk space (< 1GB)")
    
    return all_good


if __name__ == "__main__":
    # 간단한 환경 검증 실행
    validate_environment()