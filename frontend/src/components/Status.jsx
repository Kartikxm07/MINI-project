export function Spinner({ label }) {
  return (
    <div className="spinner-wrap" role="status">
      <div className="spinner" />
      {label && <span className="spinner-label">{label}</span>}
    </div>
  )
}

export function ErrorBox({ message }) {
  return (
    <div className="errorbox" role="alert">
      <strong>Something went wrong.</strong> {message}
    </div>
  )
}

export function Empty({ message }) {
  return <div className="empty">{message}</div>
}
