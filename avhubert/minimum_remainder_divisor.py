import sys

def divisores_minimo_resto(prime_number):
    if prime_number < 2:
        raise ValueError("El número debe ser mayor o igual a 2.")
    
    min_remainder = float('inf')
    divisors_with_min_remainder = []

    for divisor in range(2, prime_number):  # Divisores desde 2 hasta prime_number - 1
        remainder = prime_number % divisor
        if remainder < min_remainder:
            min_remainder = remainder
            divisors_with_min_remainder = [divisor]  # Reiniciar lista con el nuevo mínimo
        elif remainder == min_remainder:
            divisors_with_min_remainder.append(divisor)  # Agregar divisores con el mismo resto mínimo

    return divisors_with_min_remainder, min_remainder

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python minimum_remainder_divisor.py <número_primo>")
        sys.exit(1)
    
    try:
        prime_number = int(sys.argv[1])
        if prime_number < 2:
            raise ValueError

        divisors, remainder = divisores_minimo_resto(prime_number)
        print(f"Los divisores que generan el menor resto para {prime_number} son {divisors}, con un resto de {remainder}.")
    except ValueError:
        print("Por favor, introduce un número primo válido mayor o igual a 2.")
