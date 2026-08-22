import { useState, type ComponentProps } from 'react'
import { Input } from 'antd'
import { EyeInvisibleOutlined, EyeOutlined } from '@ant-design/icons'
import './AccessiblePasswordInput.css'

export type AccessiblePasswordInputProps = ComponentProps<typeof Input> & {
  showPasswordLabel: string
  hidePasswordLabel: string
}

export function AccessiblePasswordInput({
  showPasswordLabel,
  hidePasswordLabel,
  disabled,
  ...inputProps
}: AccessiblePasswordInputProps) {
  const [visible, setVisible] = useState(false)

  return (
    <Input
      {...inputProps}
      disabled={disabled}
      type={visible ? 'text' : 'password'}
      suffix={
        <button
          type="button"
          className="auth-password-toggle"
          aria-label={visible ? hidePasswordLabel : showPasswordLabel}
          aria-pressed={visible}
          disabled={disabled}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => setVisible((current) => !current)}
        >
          {visible ? <EyeInvisibleOutlined /> : <EyeOutlined />}
        </button>
      }
    />
  )
}
